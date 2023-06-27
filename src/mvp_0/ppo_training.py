from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import load_dataset
from transformers import Adafactor, AutoTokenizer

from trl import set_seed, PPOTrainer
from trl.core import LengthSampler

from watermark import watermark

from configs import get_script_args, get_ppo_config
from dataset import get_dataset, collator
from reward_model import get_reward_model
from model import get_model
from trainer import train
from utils import get_tokenizer


def main():
    info = watermark(
        packages="torch,transformers,datasets,peft,trl,tensorboard,accelerate",
        python=True, conda=True, gpu=True,
        current_date=True, current_time=True,
    )
    print(info)

    # Get script args
    script_args = get_script_args()

    # Get PPO config
    config = get_ppo_config(script_args)

    # Tokenizer
    tokenizer = get_tokenizer(script_args.tokenizer_name)

    # Dataset for PPO training
    train_dataset, prompt_max_len, response_max_len = get_dataset(
        script_args.dataset_name, tokenizer,
    )   

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    current_device = Accelerator().local_process_index
    print(f"Current device: {current_device}\n")

    # Get the reward model
    reward_model = get_reward_model(
        script_args.reward_model_output_path, current_device,
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
    output_min_length = 1
    # TODO: think about increasing this
    output_max_length = response_max_len
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    return
    train(ppo_trainer, sentiment_pipe, tokenizer, output_length_sampler, sent_kwargs, generation_kwargs, script_args, config)


if __name__ == "__main__":
    main()