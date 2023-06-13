import torch
from accelerate import Accelerator
from datasets import load_dataset
from transformers import Adafactor, AutoTokenizer

from trl import set_seed, PPOTrainer
from trl.core import LengthSampler

from watermark import watermark

from configs import get_script_args, get_ppo_config
from dataset import preprocess_dataset, collator
from reward_model import get_reward_model
from model import get_model
from trainer import train


def main():
    info = watermark(
        packages="torch,transformers,datasets,peft,trl,tensorboard,accelerate",
        python=True, conda=True, gpu=True,
        current_date=True, current_time=True,
    )
    # print(info)

    script_args = get_script_args()

    # PPO config
    config = get_ppo_config(script_args)

    # Dataset for PPO training
    train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train")
    train_dataset = train_dataset.select(range(100000))
    train_dataset = preprocess_dataset(train_dataset, tokenizer)

    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
    # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
    # only for this model.
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    current_device = Accelerator().local_process_index

    # Get the reward model
    sentiment_pipe = get_reward_model(script_args, ppo_trainer, tokenizer, current_device)

    # Model
    model = get_model(config, current_device)

    # Optimizer
    optimizer = None
    if script_args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
        )
    
    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    # TODO: put this into config
    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    sent_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 16,
        "truncation": True,
    }

    # TODO: put this into config
    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        # "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": 100_000,
    }
    output_min_length = 32
    output_max_length = script_args.output_max_length
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    train(ppo_trainer, sentiment_pipe, tokenizer, output_length_sampler, sent_kwargs, generation_kwargs, script_args, config)


if __name__ == "__main__":
    main()