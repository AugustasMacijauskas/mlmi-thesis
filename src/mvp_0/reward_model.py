import sys
from pathlib import Path

import pandas as pd

import torch
from torch import nn

from transformers import T5ForConditionalGeneration

from utils import get_model_loading_kwargs

# Add local dependencies to PATH
ELK_PATH = Path("/fsx/home-augustas/elk/")
modules = [
    ELK_PATH,
    ELK_PATH / "elk" / "training",
]
for module in modules:
    if not str(module) in sys.path:
        sys.path.insert(0, str(module.resolve()))

# Should now be possible to load the local modules
from reporter import Reporter


class MyRewardModel(nn.Module):
    def __init__(
            self, language_model, reporter,
            layer=-1, # last layer by default
            hidden_state_name="decoder_hidden_states", # decoder states by default
        ):
        super().__init__()

        # Store the language model and the reporter
        self.language_model = language_model
        self.reporter = reporter

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
        pos_logits = self.reporter(pos_last_tokens)
        neg_logits = self.reporter(neg_last_tokens)

        # Return the difference in logits which will later be
        # passed through a sigmoid function
        return pos_logits - neg_logits


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
def get_reporter_path(output_path):
    # Get the run id
    run_id = output_path.name.split("_")[-1]

    # Get the line with the elk logs from the output file
    with open(output_path / f"out.{run_id}", 'r') as file:
        vinc_logs_path = file.readlines()[-1].strip()
    vinc_logs_path = Path(vinc_logs_path.split(" ")[-1][4:-4])

    # Get the best layer
    df = pd.read_csv(vinc_logs_path / "eval.csv")

    max_value = df["acc_estimate"].max()
    max_value_rows = df[df["acc_estimate"] == max_value]

    results = []
    for _, row in max_value_rows.iterrows():
        entry = row[["layer", "ensembling"]].to_dict()
        entry["value"] = row["acc_estimate"]
        results.append(entry)

    layer = results[-1]["layer"] # best layer

    # The reporter path
    reporter_path = vinc_logs_path / "reporters" / f"layer_{layer}.pt"

    return reporter_path, layer


def get_reward_model(output_path, current_device):
    print(f"The current device is {current_device}.\n")

    # Cast to Path
    output_path = Path(output_path)

    # Get the reward model name
    language_reward_model_name = get_model_name(output_path)
    print(f"Loading reward model from {language_reward_model_name}.")
    # Check the dtype to load in
    kwargs = get_model_loading_kwargs(language_reward_model_name)

    # Load the language reward model
    # TODO: maybe make this more general to support other model classes
    model = T5ForConditionalGeneration.from_pretrained(
        language_reward_model_name, **kwargs
    ).to(current_device)
    model.eval();
    print(f"Loaded reward model with {sum(p.numel() for p in model.parameters()):,d} parameters.")
    print(f"Reward model dtype: {model.lm_head.weight.dtype}\n")

    # Get the reporter path
    reporter_path, layer = get_reporter_path(output_path)
    # Load the reporter
    print(f"Loading reporter from {reporter_path}")
    reporter = Reporter.load(reporter_path).to(current_device)
    reporter.eval()
    print("Loaded reporter.\n")

    return MyRewardModel(model, reporter, layer=layer), language_reward_model_name


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