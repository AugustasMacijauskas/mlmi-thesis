import json
import os
import sys

import torch

import datasets
datasets.disable_progress_bar() # Don't show progress datasets bars

from pathlib import Path
sys.path.insert(0, str(Path("/fsx/home-augustas/trlx/").resolve()))
import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

from model import get_model_trlx
from dataset import get_dataset_trlx
from reward_model import get_template, get_reward_model, create_reward_fn_trlx
from utils import get_tokenizer


default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=10000,
        total_steps=10000,
        batch_size=8,
        minibatch_size=None,
        tracker="wandb",
        project_name="mlmi_thesis",
        logging_dir="/fsx/home-augustas/logs_trlx",
        checkpoint_interval=10000,
        checkpoint_dir="checkpoints/ppo_hh",
        eval_interval=500,
        pipeline="PromptPipeline", # data pipeline
        trainer="AcceleratePPOTrainer",
    ),
    model=ModelConfig(model_path="gpt2-xl"),
    tokenizer=TokenizerConfig(tokenizer_path="gpt2-xl", truncation_side="left"),
    # optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    optimizer=OptimizerConfig(name="adam", kwargs=dict(lr=1e-5, betas=(0.9, 0.999), eps=1e-8)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=8e-6)),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=64,
        chunk_size=8,
        ppo_epochs=4,
        # init_kl_coef=0.05,
        init_kl_coef=0.2,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        # vf_coef=1,
        vf_coef=0.1,
        scale_reward="running",
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        # gen_kwargs=dict(
        #     max_new_tokens=128,
        #     top_k=0,
        #     top_p=1.0,
        #     do_sample=True,
        # ),
        gen_kwargs ={
            "top_k": 0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": 50256, # equals to eos_token_id
            "eos_token_id": 100_000, # why is this value like this?
            # "pad_to_multiple_of": 8, # TODO: double-check, but this seems to work and to be faster
            "max_new_tokens": 4,
        },
    ),
)


def get_reward_function(
        reward_model_output_path, template,
        rm_batch_size=96, delta_reward=False
    ):

    if not os.environ.get("RANK", "0") == "0":
        return True

    # Reward model will be on the last device
    reward_device = torch.cuda.device_count() - 1
    reward_device = torch.device(f"cuda:{reward_device}")

    # Get the reward model
    reward_model, reward_model_name = get_reward_model(
        reward_model_output_path, reward_device,
        supervised=True,
    )
    reward_model_tokenizer = get_tokenizer(reward_model_name)

    # Create the reward function
    reward_fn = create_reward_fn_trlx(
        reward_model, reward_model_tokenizer,
        rm_batch_size,
        template, reward_device,
        delta_reward=delta_reward,
    )

    return reward_fn


def main(hparams=None):
    if hparams is None:
        hparams = {}

    # device = os.environ.get("RANK")
    # device = torch.device(f"cuda:{device}")
    # model = get_model_trlx("gpt2-xl", device)
    # hparams = { "model": { "model_path": model } }

    config = TRLConfig.update(default_config, hparams)

    # Dataset
    dataset_dict = get_dataset_trlx(
        dataset_name="AugustasM/burns-datasets-VINC-imdb-ppo-training-v2",
        train_dataset_size=8192,
        eval_dataset_size=512,
    )
    prompts = dataset_dict["train"].to_list()
    eval_prompts = dataset_dict["eval"].to_list()
    print(f"{len(prompts)=}")
    print(f"{len(eval_prompts)=}")


    # Dataset templates
    dataset_template_path = "AugustasM/burns-datasets-VINC"
    template = get_template(dataset_template_path)


    # Create reward function
    reward_model_output_path = "/fsx/home-augustas/logs/UQA-3b-custom_data_imdb_v2_final_20230717_200713_36998"
    # reward_model_output_path = "/fsx/home-augustas/logs/UQA-11b-custom_data_imdb_v2_final_20230718_172607_37534"

    rm_batch_size = 64 if "3b" in reward_model_output_path else 16
    reward_fn = get_reward_function(
        reward_model_output_path=reward_model_output_path,
        template=template,
        rm_batch_size=rm_batch_size,
        delta_reward=False,
    )


    # Train
    trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=reward_fn,
        config=config,
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)