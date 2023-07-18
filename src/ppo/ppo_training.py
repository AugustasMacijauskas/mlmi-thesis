from pathlib import Path

import torch

import datasets

from accelerate import Accelerator

from trl import set_seed, PPOTrainer

from watermark import watermark

from datetime import datetime

from configs import get_script_args, get_ppo_config
from dataset import get_dataset, collator
from reward_model import get_reward_model, create_reward_fn
from model import get_model
from trainer import train
from utils import get_tokenizer, get_template


def main():
    # Don't show progress datasets bars
    datasets.disable_progress_bar()

    info = watermark(
        packages="torch,transformers,datasets,peft,trl,tensorboard,accelerate",
        python=True, conda=True, gpu=True,
        current_date=True, current_time=True,
    )
    print(info)

    # Get script args
    script_args = get_script_args()
    print(script_args)

    # Get PPO config
    config = get_ppo_config(script_args)
    print(f"{config.total_ppo_epochs=}")

    # Tokenizer
    tokenizer = get_tokenizer(script_args.tokenizer_name)

    # Dataset for PPO training
    train_dataset, prompt_max_len, response_max_len = get_dataset(
        script_args.dataset_name, tokenizer, subsets_to_delete=["piqa"]
    )

    # Dataset templates
    dataset_template_path = "AugustasM/burns-datasets-VINC"
    template = get_template(dataset_template_path)

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    current_device = Accelerator().local_process_index
    current_device = torch.device(f"cuda:{current_device}")
    # print(f"Current device: {current_device}\n")

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
    model = get_model(script_args.model_name, current_device)

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
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": 100_000, # why is this value like this?
        "pad_to_multiple_of": 8, # TODO: double-check, but this seems to work and to be faster
        # "max_new_tokens": response_max_len, # TODO: think how to approach this
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
    print(f"Duration: {elapsed_time}")


if __name__ == "__main__":
    main()