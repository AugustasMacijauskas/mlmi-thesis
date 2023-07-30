from peft import LoraConfig

try:
    from trl import AutoModelForCausalLMWithValueHead
except ModuleNotFoundError:
    from transformers import AutoModelForCausalLM

from utils import get_model_loading_kwargs, freeze_bottom_causal_layers

from accelerate import Accelerator
temporary_accelerator = Accelerator()


def get_model_with_lora(model_name, device, lora_config, **kwargs):
    # TODO: add LoRA configuration to the config object and not hard-coded
    temporary_accelerator.print("Loading policy model...\n")

    temporary_accelerator.print(f"{kwargs=}")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        device_map={ "": device },
        load_in_8bit=True,
        peft_config=lora_config,
        **kwargs,
    )

    temporary_accelerator.print(f"Loaded subject model with {sum(p.numel() for p in model.parameters()):,d} parameters.")
    temporary_accelerator.print(f"Number of trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad):,d} parameters.")
    temporary_accelerator.print(f"Model dtype: {next(iter(model.parameters())).dtype}\n")

    return model


def get_model_with_unfrozen_layers(model_name, device, num_layers_unfrozen=3, **kwargs):
    # TODO: add LoRA configuration to the config object and not hard-coded
    temporary_accelerator.print("Loading policy model...\n")

    temporary_accelerator.print(f"{kwargs=}")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        device_map={ "": device },
        # load_in_8bit=True,
        **kwargs,
    )

    # Make the whole model trainable before freezing the bottom layers
    model.pretrained_model.requires_grad_(True)
    model.pretrained_model.model.embed_tokens.requires_grad_(False)

    temporary_accelerator.print(f"Unfreezing bottom {num_layers_unfrozen} layers...")
    freeze_bottom_causal_layers(
        model.pretrained_model, num_layers_unfrozen=num_layers_unfrozen
    )

    temporary_accelerator.print(f"Loaded subject model with {sum(p.numel() for p in model.parameters()):,d} parameters.")
    temporary_accelerator.print(f"Number of trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad):,d} parameters.")
    temporary_accelerator.print(f"Model dtype: {next(iter(model.parameters())).dtype}\n")

    return model


def get_model(model_name, current_device, **kwargs):
    temporary_accelerator.print("Loading policy model...\n")

    kwargs, _ = get_model_loading_kwargs(model_name, **kwargs)
    temporary_accelerator.print(f"{kwargs=}")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        device_map={ "": current_device },
        **kwargs,
    )

    temporary_accelerator.print(f"Loaded subject model with {sum(p.numel() for p in model.parameters()):,d} parameters.")
    temporary_accelerator.print(f"Model dtype: {next(iter(model.parameters())).dtype}\n")

    return model


def get_model_trlx(model_name, device):
    temporary_accelerator.print("Loading policy model...\n")

    kwargs, _ = get_model_loading_kwargs(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # load_in_8bit=True,
        device_map={ "": device },
        **kwargs,
    )

    temporary_accelerator.print(f"Loaded subject model with {sum(p.numel() for p in model.parameters()):,d} parameters.")
    temporary_accelerator.print(f"Model dtype: {next(iter(model.parameters())).dtype}\n")

    return model