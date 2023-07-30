import functools
from typing import Tuple, Union

import torch

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    AutoConfig,
)

from accelerate import Accelerator
temporary_accelerator = Accelerator()


def get_tokenizer(model_str: str, **kwargs) -> PreTrainedTokenizerBase:
    temporary_accelerator.print(f"Loading tokenizer {model_str}...")

    """Instantiate a tokenizer, using the fast one iff it exists."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_str, use_fast=True, **kwargs)
    
    except Exception as e:
        if kwargs.get("verbose", True):
            temporary_accelerator.print(f"Falling back to slow tokenizer; fast one failed: '{e}'")

        tokenizer = AutoTokenizer.from_pretrained(model_str, use_fast=False, **kwargs)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # For some reason the original tokenizer has gibberish for the
    # max length of the model, so we set it manually here
    if "dolly-v2" in model_str:
        tokenizer.model_max_length = 2048
    
    temporary_accelerator.print("Loaded tokenizer.\n")

    return tokenizer


def get_model_loading_kwargs(model_name, **kwargs):
    model_cfg = AutoConfig.from_pretrained(model_name)
    fp32_weights = model_cfg.torch_dtype in (None, torch.float32)
    bf16_weights = model_cfg.torch_dtype == torch.bfloat16
    is_bf16_possible = (bf16_weights or fp32_weights) and torch.cuda.is_bf16_supported()
    temporary_accelerator.print(f"{is_bf16_possible=}")

    if kwargs.get("load_in_8bit") and not fp32_weights:
        kwargs["torch_dtype"] = None
    
    elif is_bf16_possible:
        kwargs["torch_dtype"] = torch.bfloat16
    
    else:
        kwargs["torch_dtype"] = "auto"
    
    return kwargs, is_bf16_possible


def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def rgetattr(obj, attr: str, *args) -> object:
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def findattr(obj, attrs: Tuple[str]) -> Union[object, None]:
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    raise ValueError(f"Could not find an attribute from `{attrs}` in `{obj}`")


def hf_get_decoder_blocks(model: torch.nn.Module) -> Tuple[torch.nn.Module]:
    """Returns the decoder hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
        - decoder.block: (T5ForConditionalGeneration)
    """
    hidden_layers_attrs = (
        "h",
        "layers",
        "model.layers",
        "decoder.layers",
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
        "decoder.block",
    )
    return findattr(model, hidden_layers_attrs)


def freeze_bottom_causal_layers(model: torch.nn.Module, num_layers_unfrozen: int = 0):
    """Freezes the bottom transformer block layers of the specified model."""
    hidden_layers = hf_get_decoder_blocks(model)
    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
    else:
        hidden_layers_to_freeze = []
    for layer in hidden_layers_to_freeze:
        layer.requires_grad_(False)


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