from pathlib import Path

import pandas as pd

import torch
from torch import nn

from datasets import Dataset

from transformers import T5ForConditionalGeneration

from utils import get_model_loading_kwargs

from pathlib import Path
import sys
ELK_PATH = Path("/fsx/home-augustas/elk/")
modules = [
    ELK_PATH,
    ELK_PATH / "elk" / "promptsource",
]
for module in modules:
    if not str(module) in sys.path:
        sys.path.insert(0, str(module.resolve()))

from templates import DatasetTemplates

from accelerate import Accelerator
temporary_accelerator = Accelerator()


def get_template(dataset_template_path):
    dataset_templates = DatasetTemplates(dataset_template_path)
    dataset_templates.templates = {
        x.name: x for x in dataset_templates.templates.values()
    }
    temporary_accelerator.print(f"Num templates: {len(dataset_templates.templates)}")
    template = list(dataset_templates.templates.values())[0]
    temporary_accelerator.print(f"{template.name}")

    return template


class MyRewardModel(nn.Module):
    def __init__(
            self, language_model, probe,
            layer=-1, # last layer by default
            hidden_state_name="decoder_hidden_states", # decoder states by default
        ):
        super().__init__()

        # Store the language model and the probe
        self.language_model = language_model
        self.probe = probe

        # Store other variables
        self.layer = layer # which layer to extract
        self.hidden_state_name = hidden_state_name
        self.pad_token_id = self.language_model.config.pad_token_id

    
    def forward(self, pos_inputs, neg_inputs):
        # Get the hidden states
        # Shape B x T x H
        pos_hidden_states = self.language_model(
            **pos_inputs, output_hidden_states=True,
        )[self.hidden_state_name][self.layer]
        neg_hidden_states = self.language_model(
            **neg_inputs, output_hidden_states=True,
        )[self.hidden_state_name][self.layer]

        pos_last_token_index = (
            pos_inputs["labels"] == self.pad_token_id
        ).int().argmax(dim=1) - 1
        neg_last_token_index = (
            neg_inputs["labels"] == self.pad_token_id
        ).int().argmax(dim=1) - 1

        # Get the last token's output
        pos_last_tokens = pos_hidden_states[range(len(pos_last_token_index)), pos_last_token_index]
        neg_last_tokens = neg_hidden_states[range(len(neg_last_token_index)), neg_last_token_index]

        # Get the logits for the two classes
        pos_logits = self.probe(pos_last_tokens)
        neg_logits = self.probe(neg_last_tokens)

        # Return the difference in logits which will later be
        # passed through a sigmoid function
        return (pos_logits - neg_logits).float()


# Utility function to get the model name from the output path
def get_model_name(output_path):
    # Get the run id
    run_id = output_path.name.split("_")[-1]

    # Get the line with the model name from the slurm output file
    with open(output_path / f"slurm-{run_id}.out", 'r') as file:
        line_with_model_name = [l.strip() for l in file.readlines() if "elk elicit" in l][0]

    # Extract the model name
    return line_with_model_name.split(" ")[2]


# Utility function to get the reporter path from the output path
def get_probe(
        output_path, current_device,
        is_bf16_possible=False,
        supervised=True, metric="auroc_estimate", ensembling="partial",
    ):
    # Get the run id
    run_id = output_path.name.split("_")[-1]

    # Get the line with the elk logs from the output file
    with open(output_path / f"out.{run_id}", 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    
    output_directory_line = [line for line in lines if line.startswith("Output directory at")]
    assert len(output_directory_line) == 1
    vinc_logs_path = Path(output_directory_line[0].split(" ")[-1][4:-4])

    # Get the best layer
    filename = "lr_eval.csv" if supervised else "eval.csv"
    df = pd.read_csv(vinc_logs_path / filename)
    df = df[df["ensembling"] == ensembling]

    max_value = df[metric].max()
    max_value_rows = df[df[metric] == max_value]

    layer = max_value_rows["layer"].values[-1] # best layer
    temporary_accelerator.print(f"{layer=}")

    # The reporter path
    folder_name = "lr_models" if supervised else "reporters"
    probe_path = vinc_logs_path / folder_name / f"layer_{layer}.pt"

    # Load the probe
    temporary_accelerator.print(f"Loading the probe from {probe_path}")
    probe = torch.load(probe_path, map_location=current_device)

    if supervised:
        assert len(probe) == 1
        probe = probe[0]
        probe.eval()
        probe.requires_grad_(False)

        if is_bf16_possible:
            probe = probe.bfloat16()
    else:
        probe.bias.requires_grad_(False)
        probe.scale.requires_grad_(False)

        if is_bf16_possible:
            probe.weight = probe.weight.bfloat16()
            probe.bias = probe.bias.bfloat16()
            probe.scale = probe.scale.bfloat16()

    temporary_accelerator.print("Finished loading the probe.\n")

    return probe, layer


def get_reward_model(
        output_path, current_device,
        **probe_kwargs,
    ):
    temporary_accelerator.print(f"The current device is {current_device}.\n")

    # Cast to Path
    output_path = Path(output_path)

    # Get the reward model name
    language_reward_model_name = get_model_name(output_path)
    temporary_accelerator.print(f"Loading reward model from {language_reward_model_name}.")
    # Check the dtype to load in
    kwargs, is_bf16_possible = get_model_loading_kwargs(language_reward_model_name)

    # Load the language reward model
    # TODO: maybe make this more general to support other model classes
    model = T5ForConditionalGeneration.from_pretrained(
        language_reward_model_name, **kwargs
    ).to(current_device)
    model.eval()
    model.requires_grad_(False)
    temporary_accelerator.print(f"Loaded reward model with {sum(p.numel() for p in model.parameters()):,d} parameters.")
    temporary_accelerator.print(f"Number of trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad):,d} parameters.")
    temporary_accelerator.print(f"Reward model dtype: {model.lm_head.weight.dtype}\n")

    # Get the reporter path
    probe, layer = get_probe(
        output_path, current_device,
        is_bf16_possible=is_bf16_possible, **probe_kwargs
    )

    return MyRewardModel(model, probe, layer=layer), language_reward_model_name


# Idea borrowed from: https://github.com/CarperAI/trlx/blob/main/examples/hh/ppo_hh.py
def create_reward_fn(
    reward_model, reward_model_tokenizer,
    rm_batch_size,
    template, device,
):
    def tokenization_function(q, a):
        return reward_model_tokenizer(
            q, text_target=a.strip(),
            add_special_tokens=True, return_tensors="pt",
        )
    
    def preprocess_function(examples):
        # Initialize the output dictionary
        processed_examples = {
            "pos_input_ids": [],
            "pos_attention_mask": [],
            "pos_labels": [],
            "neg_input_ids": [],
            "neg_attention_mask": [],
            "neg_labels": [],
        }

        for text in examples["text"]:
            entry = { "text": text }

            # Get the positive and negative examples
            entry["label"] = 1
            pos_q, pos_a = template.apply(entry)

            entry["label"] = 0
            neg_q, neg_a = template.apply(entry)

            # Tokenize the inputs
            pos_inputs = tokenization_function(pos_q, pos_a)
            neg_inputs = tokenization_function(neg_q, neg_a)

            # Append the results to the corresponding lists in the output dictionary
            for key in processed_examples.keys():
                if key.startswith('pos'):
                    processed_examples[key].append(pos_inputs[key.replace('pos_', '')].squeeze().tolist())
                else:
                    processed_examples[key].append(neg_inputs[key.replace('neg_', '')].squeeze().tolist())

        return processed_examples
    

    def prepare_batch(texts):
        text_dataset = Dataset.from_dict({ "text": texts })

        processed_dataset = text_dataset.map(
            preprocess_function, batched=True,
            remove_columns=text_dataset.column_names,
        )

        # Get max length
        max_length = max([len(item) for item in processed_dataset["pos_input_ids"]])

        # Pad pos features
        features_to_pad = {
            "input_ids": processed_dataset["pos_input_ids"],
            "attention_mask": processed_dataset["pos_attention_mask"],
        }
        pos_inputs = reward_model_tokenizer.pad(
            features_to_pad, padding="longest", return_tensors="pt",
        ).to(device)
        assert all([item.shape[1] == max_length for item in pos_inputs.values()])
        pos_inputs["labels"] = torch.LongTensor(processed_dataset["pos_labels"]).to(device)

        # Pad neg features
        features_to_pad = {
            "input_ids": processed_dataset["neg_input_ids"],
            "attention_mask": processed_dataset["neg_attention_mask"],
        }
        neg_inputs = reward_model_tokenizer.pad(
            features_to_pad, padding="longest", return_tensors="pt",
        ).to(device)
        assert all([item.shape[1] == max_length for item in neg_inputs.values()])
        neg_inputs["labels"] = torch.LongTensor(processed_dataset["neg_labels"]).to(device)

        return pos_inputs, neg_inputs

    
    def get_rewards(texts):
        '''
            args:
                texts: list of strings - the prompts concatenated with the generations
        '''

        rewards = []

        # Batching
        for i in range(0, len(texts), rm_batch_size):
            text_batch = texts[i:i+rm_batch_size]
            
            pos_inputs, neg_inputs = prepare_batch(text_batch)

            with torch.no_grad():
                predictions = reward_model(pos_inputs, neg_inputs)

            rewards.append(predictions.sigmoid())
        
        return torch.cat(rewards, dim=0)
    
    return get_rewards


def create_reward_fn_trlx(
        reward_model, reward_model_tokenizer,
        rm_batch_size,
        template, device,
        delta_reward=True,
    ):

    get_rewards = create_reward_fn(
        reward_model, reward_model_tokenizer,
        rm_batch_size,
        template, device,
    )

    def reward_fn(samples, prompts, original_output, **kwargs):
        rewards = get_rewards(samples)

        if not delta_reward:
            return rewards

        original_samples = [p + o for p, o in zip(prompts, original_output)]
        original_rewards = get_rewards(original_samples)
        
        return rewards - original_rewards

    return reward_fn


def main():
    output_path = (
        "/fsx/home-augustas/logs_old/"
        "unifiedqa-v2-t5-3b-1363200_custom_data_all_20230622_180051_15555"
    )
    reward_model = get_reward_model(
        output_path, current_device="cuda:0",
    )


if __name__ == "__main__":
    main()