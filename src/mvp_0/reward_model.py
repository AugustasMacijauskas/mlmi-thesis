import sys
from pathlib import Path

import torch
from torch import nn

from transformers import AutoConfig, T5ForConditionalGeneration

# Add local dependencies to PATH
ELK_PATH = Path("/fsx/home-augustas/elk/")
modules = [
    ELK_PATH,
    ELK_PATH / "elk" / "training",
    ELK_PATH / "elk" / "promptsource",
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


def get_reward_model(
        language_reward_model_name,
        reporter_dir, layer, current_device,
    ):
    print(f"The current device is {current_device}.")

    # Check the dtype to load in
    model_cfg = AutoConfig.from_pretrained(language_reward_model_name)
    fp32_weights = model_cfg.torch_dtype in (None, torch.float32)
    is_bf16_possible = fp32_weights and torch.cuda.is_bf16_supported()
    print(f"{is_bf16_possible=}")
    kwargs = {
        "torch_dtype": torch.bfloat16 if is_bf16_possible else torch.float32
    }

    # Load the language reward model
    print(f"Loading reward model from {language_reward_model_name}.")
    model = T5ForConditionalGeneration.from_pretrained(
        language_reward_model_name, **kwargs
    ).to(current_device)
    model.eval();
    print(f"Loaded reward model with {sum(p.numel() for p in model.parameters()):,d} parameters.")
    print(f"Reward model dtype: {model.lm_head.weight.dtype}")

    # Load the reporter
    reporter_path = reporter_dir / "reporters" / f"layer_{layer}.pt"
    print(f"Loading reporter from {reporter_path}.")
    reporter = Reporter.load(reporter_path).to(current_device)
    reporter.eval()
    print("Loaded reporter.")

    return MyRewardModel(model, reporter, layer=layer)


def main():
    reporter_dir = (
        "/fsx/home-augustas/VINC-logs/"
        "allenai/unifiedqa-v2-t5-3b-1363200/"
        "AugustasM/burns-datasets-VINC/sad-carson"
    )
    reporter_dir = Path(reporter_dir)
    reward_model = get_reward_model(
        language_reward_model_name="allenai/unifiedqa-v2-t5-3b-1363200",
        reporter_dir=reporter_dir,
        layer=18, current_device="cuda:0",
    )


if __name__ == "__main__":
    main()