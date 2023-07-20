from pathlib import Path

import torch

import datasets
# Don't show progress datasets bars
datasets.disable_progress_bar()

import sys
sys.path.insert(0, str(Path("/fsx/home-augustas/mlmi-thesis/src/ppo").resolve()))
from dataset import get_dataset_trlx
# from model import get_model
from reward_model import get_template, get_reward_model, create_reward_fn_trlx
from utils import get_tokenizer


device = torch.device(
    f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
)
print(f"{device=}")


tokenizer_name = "gpt2-xl"

dataset_name = "AugustasM/burns-datasets-VINC-imdb-ppo-training-v2"
# reward_model_output_path = "/fsx/home-augustas/logs/UQA-3b-custom_data_imdb_v2_final_20230717_200713_36998"
reward_model_output_path = "/fsx/home-augustas/logs/UQA-11b-custom_data_imdb_v2_final_20230718_172607_37534"


dataset = get_dataset_trlx(dataset_name, eval_dataset_size=280)

prompts = dataset["train"].to_list()
eval_prompts = dataset["eval"].to_list()

print(f"{len(prompts)=}")
print(f"{len(eval_prompts)=}")


# Dataset templates
dataset_template_path = "AugustasM/burns-datasets-VINC"
template = get_template(dataset_template_path)


# Get the reward model
reward_model, reward_model_name = get_reward_model(
    reward_model_output_path, device,
    supervised=True,
)
reward_model_tokenizer = get_tokenizer(reward_model_name)


rm_batch_size = 16
print(f"{rm_batch_size=}")
prompts = dataset["train"]["prompt"][:rm_batch_size]
responses = dataset["train"]["original_output"][:rm_batch_size]
texts = [q + r for q, r in zip(prompts, responses)]

print(f"{len(texts)=}")
print(texts[0])


# Create reward function
reward_fn = create_reward_fn_trlx(
    reward_model=reward_model,
    reward_model_tokenizer=reward_model_tokenizer,
    rm_batch_size=rm_batch_size,
    template=template,
    device=device,
    delta_reward=False,
)

rewards = reward_fn(texts, prompts, responses)

print(rewards.shape)
print(rewards)
