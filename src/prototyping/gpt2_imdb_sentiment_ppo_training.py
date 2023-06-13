from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from watermark import watermark


tqdm.pandas()


# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={ "project_dir": PATH_TO_LOGS }` to the PPOConfig.


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(
        # default="edbeeching/gpt-neo-125M-imdb",
        default="gpt2",
        metadata={ "help": "name of the model that will be optimized with PPO" }
    )
    log_with: Optional[str] = field(default=None, metadata={ "help": "use 'wandb'/'tensorboard' to log" })
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={ "help": "the learning rate" })
    mini_batch_size: Optional[int] = field(default=16, metadata={ "help": "the PPO minibatch size" })
    batch_size: Optional[int] = field(default=256, metadata={ "help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={ "help": "the number of gradient accumulation steps" }
    )


def get_ppo_config():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    return PPOConfig(
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        mini_batch_size=script_args.mini_batch_size,
        batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        accelerator_kwargs={ "project_dir": "tensorboard" }
    )


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(tokenizer, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def get_model(config):
    # TODO: add LoRA configuration to the config object and not hard-coded
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # TODO: add support for 8 bit mode
    return AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        # load_in_8bit=True,
        peft_config=lora_config,
    )


# TODO: this comment is misplaced
# Apply LoRA
# Here comes the magic with `peft`!
# Let's load a `PeftModel` and specify that
# we are going to use low-rank adapters (LoRA)
# using `get_peft_model` utility function from `peft`.
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    print((
        f"Trainable PEFT parameters: {trainable_params / 1e6:.2f} M/"
        f"{all_param / 1e6:.2f} M "
        f"({100 * trainable_params / all_param:.2f}%)"
    ))


def get_reward_model(config, ppo_trainer, model):
    # We then build the sentiment analysis pipeline, passing the model name and the
    # sentiment analysis pipeline arguments. Let's also make sure to set the device
    # to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    print("Device:", device)
    if ppo_trainer.accelerator.num_processes == 1:
        device = model.current_device if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

    return sentiment_pipe


def train(ppo_trainer, model, tokenizer, sentiment_pipe, config):
    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    sent_kwargs = {
        "top_k": None,
        "function_to_apply": "none",
        "batch_size": config.mini_batch_size,
    }

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": -1,
    }
    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)


    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        print(epoch)
        query_tensors = batch["input_ids"]

        # cache and gradient checkpointing are not compatible, so we switch them on and off here
        model.gradient_checkpointing_disable()
        model.pretrained_model.config.use_cache = True
        # Get response from Causal LM
        response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, length_sampler=output_length_sampler, **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)

        # Compute sentiment score
        # TODO: the texts won't be aligned, which is probably why `sentiment_pipe` is slow
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        # Run PPO step
        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    # model.push_to_hub(f"{config.model_name}-ppo-sentiment")



def main():
    info = watermark(
        packages="torch,transformers,datasets,peft,trl,tensorboard",
        python=True, conda=True, gpu=True,
        current_date=True, current_time=True,
    )
    # print(info)

    config = get_ppo_config()

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # We retrieve the dataset by calling the `build_dataset` function.
    dataset = build_dataset(tokenizer)

    # Load the model
    model = get_model(config)
    print_trainable_parameters(model)

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config, model, ref_model=None,
        tokenizer=tokenizer, dataset=dataset, data_collator=collator
    )

    sentiment_pipe = get_reward_model(config, ppo_trainer, model)

    train(ppo_trainer, model, tokenizer, sentiment_pipe, config)



if __name__ == "__main__":
    main()
