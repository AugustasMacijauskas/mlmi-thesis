from tqdm import tqdm

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()

        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6b", cache_dir="../../../.hf_cache/hub"
        )

        self.transformer = model.transformer
        self.v_head = nn.Linear(model.config.n_embd, 1, bias=False, dtype=torch.float16)
        
        self.PAD_ID = model.config.eos_token_id


    def forward(self, input_ids, attention_mask):
        hidden_states = self.transformer(
            input_ids, attention_mask=attention_mask
        )[0]


        rewards = self.v_head(hidden_states).squeeze(-1)
        
        ends = (input_ids == self.PAD_ID).int().argmax(dim=1, keepdim=True)
        rewards = torch.gather(rewards, 1, ends)
        
        return rewards
    

def load_model(weights_path, device):
    state_dict = torch.load(weights_path)

    reward_model = RewardModel()
    reward_model.load_state_dict(state_dict, strict=False)
    reward_model.to(device)
    reward_model.v_head.weight.data = reward_model.v_head.weight.data.float()

    reward_model.eval()

    return reward_model


def eval_model(
        reward_model, tokenizer, dataset,
        device, max_length, eos_token, batch_size=32
    ):
    
    # Create a data loader
    dataloader = DataLoader(dataset, batch_size=batch_size)

    list_of_rewards = []

    # Iterate over batches and perform inference
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        
        prompts = batch["prompt"]
        chosen_responses = batch["chosen"]
        
        chosen_responses = [
            prompt + chosen_response + eos_token for prompt, chosen_response in zip(prompts, chosen_responses)
        ]

        # Note that we do not have to truncate in this specific case since
        # all input sequences will be shorter than max_length
        chosen_tokenized = tokenizer(
            chosen_responses,
            padding="max_length", max_length=max_length, return_tensors="pt",
        )
        
        with torch.no_grad():
            rewards = reward_model(
                input_ids=chosen_tokenized["input_ids"].to(device),
                attention_mask=chosen_tokenized["attention_mask"].to(device),
            )
        
        list_of_rewards.append(rewards.squeeze(dim=-1))

        # if i == 9: break

    return torch.cat(list_of_rewards).cpu()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = (
        "../../../.hf_cache/hub/models--Dahoas--gptj-rm-static/"
        "snapshots/dc9bb2f15f4cddace8a812174c3e7afda2308258/hf_ckpt.pt"
    )
    reward_model = load_model(weights_path, device=device)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
    tokenizer.pad_token = tokenizer.eos_token

    MAX_LENGTH = 1024
    EOS_TOKEN = tokenizer.eos_token

    dataset = load_dataset(
        "Dahoas/rm-static", split="test",
        cache_dir="../../../.hf_cache/datasets"
    )

    all_rewards = eval_model(
        reward_model, tokenizer, dataset,
        max_length=MAX_LENGTH, eos_token=EOS_TOKEN,
        device=device, batch_size=32,
    )
    
    df = pd.DataFrame(all_rewards, columns=["value"])
    df.to_csv("reward_distributions.csv", index=False)


if __name__ == "__main__":
    main()

