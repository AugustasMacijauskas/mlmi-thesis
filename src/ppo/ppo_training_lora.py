from datetime import datetime

from watermark import watermark
import torch
import datasets
from accelerate import Accelerator
from trl import set_seed, PPOTrainer
from peft import LoraConfig

from configs import get_script_args, get_ppo_config
from dataset import get_dataset_qnli, collator
from reward_model import get_template, get_reward_model, create_reward_fn
from model import get_model_with_lora
from trainer import train
from utils import get_tokenizer


def main():
    # Don't show progress datasets bars
    datasets.disable_progress_bar()

    temporary_accelerator = Accelerator()

    info = watermark(
        packages="torch,transformers,datasets,peft,trl,tensorboard,accelerate",
        python=True, conda=True, gpu=True,
        current_date=True, current_time=True,
    )
    temporary_accelerator.print(info)

    # Get script args
    script_args = get_script_args()
    temporary_accelerator.print(script_args)

    # Get PPO config
    config = get_ppo_config(script_args)
    temporary_accelerator.print(f"{config.total_ppo_epochs=}")

    # Tokenizer
    tokenizer = get_tokenizer(script_args.tokenizer_name)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.unk_token

    # Dataset for PPO training
    train_dataset = get_dataset_qnli(
        script_args.dataset_name, tokenizer, margin=8,
        num_examples=81920, seed=config.seed,
    )

    # Dataset templates
    dataset_template_path = "AugustasM/burns-datasets-VINC"
    template = get_template(dataset_template_path)

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    current_device = Accelerator().local_process_index
    current_device = torch.device(f"cuda:{current_device}")

    # Get the reward model
    reward_model, reward_model_name = get_reward_model(
        script_args.reward_model_output_path, current_device,
        supervised=True,
    )
    reward_model_tokenizer = get_tokenizer(reward_model_name)

    # Create reward function
    get_rewards = create_reward_fn(
        reward_model=reward_model,
        reward_model_tokenizer=reward_model_tokenizer,
        rm_batch_size=script_args.rm_batch_size,
        template=template,
        device=current_device,
    )

    # Model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_model_with_lora(
        script_args.model_name, current_device, lora_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    memory_usage = model.pretrained_model.get_memory_footprint() / (1024 ** 3)
    temporary_accelerator.print(f"{memory_usage=:.2f} GB")

    # Optimizer
    # TODO: consider whether adding Adafactor back in is a good idea
    optimizer = None

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
    # TODO: check whether there are better settings for this
    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "top_k": 0,
        "top_p": 1.0,
        # "do_sample": False,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": 100_000, # why is this value like this?
        "pad_to_multiple_of": 8, # TODO: double-check, but this seems to work and to be faster
        "max_new_tokens": 1,
    }

    # Store starting time to get training duration in HH:MM:SS format later
    start_time = datetime.now()
    train(
        ppo_trainer=ppo_trainer,
        tokenizer=tokenizer,
        generation_kwargs=generation_kwargs,
        get_rewards=get_rewards,
        script_args=script_args,
        config=config,
    )
    elapsed_time = datetime.now() - start_time
    elapsed_time = datetime.utcfromtimestamp(
        elapsed_time.total_seconds()
    ).strftime("%H:%M:%S")
    ppo_trainer.accelerator.print(f"Duration: {elapsed_time}")


if __name__ == "__main__":
    main()