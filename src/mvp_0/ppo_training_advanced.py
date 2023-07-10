import os

rank = os.environ.get("RANK", "0") 
print(f"{rank=}")
print(rank == "0")

from pathlib import Path

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
from utils import get_tokenizer

import sys
ELK_PATH = Path("/fsx/home-augustas/elk/")
modules = [
    ELK_PATH,
    ELK_PATH / "elk" / "promptsource",
]
for module in modules:
    if not str(module) in sys.path:
        sys.path.insert(0, str(module.resolve()))

print(sys.path[:2])

from templates import DatasetTemplates


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
    dataset_template_path = "AugustasM/burns-datasets-VINC/first"
    dataset_templates = DatasetTemplates(dataset_template_path)
    dataset_templates.templates = {
        x.name: x for x in dataset_templates.templates.values()
    }
    print(f"Num templates: {len(dataset_templates.templates)}")
    template = list(dataset_templates.templates.values())[0]

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    current_device = Accelerator().local_process_index
    # print(f"Current device: {current_device}\n")

    # Get the reward model
    # Approach: only the process with rank 0 loads the reward model,
    # but later it is moved to only the last process.
    if os.environ.get("RANK", "0") == "0":
        reward_model, reward_model_name = get_reward_model(
            script_args.reward_model_output_path, current_device,
        )
        reward_model_tokenizer = get_tokenizer(reward_model_name)

        # Move the reward model to the last process
        reward_model_device = script_args.num_gpus - 1
        print(f"Moving reward model to device {reward_model_device}")
        reward_model = reward_model.to(reward_model_device)

        # Create reward function
        get_rewards = create_reward_fn(
            reward_model, reward_model_tokenizer,
            script_args.rm_batch_size,
            template, reward_model_device,
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
    }

    # Store starting time to get training duration in HH:MM:SS format later

    start_time = datetime.now()
    train(
        ppo_trainer=ppo_trainer,
        tokenizer=tokenizer,
        generation_kwargs=generation_kwargs,
        get_rewards=get_rewards,
        template=template,
        script_args=script_args,
        config=config,
        device=current_device,
    )
    elapsed_time = datetime.now() - start_time
    elapsed_time = datetime.utcfromtimestamp(
        elapsed_time.total_seconds()
    ).strftime("%H:%M:%S")
    print(f"Duration: {elapsed_time}")


if __name__ == "__main__":
    main()