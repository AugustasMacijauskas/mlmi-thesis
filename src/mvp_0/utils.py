import torch

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    AutoConfig,
)


def get_tokenizer(model_str: str, **kwargs) -> PreTrainedTokenizerBase:
    print(f"Loading tokenizer {model_str}...")

    """Instantiate a tokenizer, using the fast one iff it exists."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_str, use_fast=True, **kwargs)
    
    except Exception as e:
        if kwargs.get("verbose", True):
            print(f"Falling back to slow tokenizer; fast one failed: '{e}'")

        tokenizer = AutoTokenizer.from_pretrained(model_str, use_fast=False, **kwargs)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loaded tokenizer.\n")

    return tokenizer


def get_model_loading_kwargs(model_name):
    model_cfg = AutoConfig.from_pretrained(model_name)
    fp32_weights = model_cfg.torch_dtype in (None, torch.float32)
    is_bf16_possible = fp32_weights and torch.cuda.is_bf16_supported()
    print(f"{is_bf16_possible=}")
    
    return {
        "torch_dtype": torch.bfloat16 if is_bf16_possible else torch.float32
    }


def main():
    model_names = [
        "EleutherAI/gpt-j-6b",
        "allenai/unifiedqa-t5-3b",
        "allenai/unifiedqa-v2-t5-3b-1363200",
        "allenai/unifiedqa-v2-t5-11b-1363200",
        "microsoft/deberta-v2-xxlarge",
        ]
    for model_name in model_names:
        tokenizer = get_tokenizer(model_name)
        print(f"{model_name}: {type(tokenizer)}")
        print()


if __name__ == "__main__":
    main()