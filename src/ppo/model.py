from peft import LoraConfig

try:
    from trl import AutoModelForCausalLMWithValueHead
except ModuleNotFoundError:
    from transformers import AutoModelForCausalLM

from utils import get_model_loading_kwargs


def get_model_with_lora(config, current_device):
    # TODO: add LoRA configuration to the config object and not hard-coded
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        load_in_8bit=True,
        device_map={ "": current_device },
        peft_config=lora_config,
    )

    return model


def get_model(model_name, current_device, **kwargs):
    print("Loading policy model...\n")

    kwargs, _ = get_model_loading_kwargs(model_name, **kwargs)
    print(f"{kwargs=}")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        device_map={ "": current_device },
        **kwargs,
    )

    print(f"Loaded subject model with {sum(p.numel() for p in model.parameters()):,d} parameters.")
    print(f"Model dtype: {next(iter(model.parameters())).dtype}\n")

    return model


def get_model_trlx(model_name, device):
    print("Loading policy model...\n")

    kwargs, _ = get_model_loading_kwargs(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # load_in_8bit=True,
        device_map={ "": device },
        **kwargs,
    )

    print(f"Loaded subject model with {sum(p.numel() for p in model.parameters()):,d} parameters.")
    print(f"Model dtype: {next(iter(model.parameters())).dtype}\n")

    return model