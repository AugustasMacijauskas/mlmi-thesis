from peft import LoraConfig

from trl import AutoModelForCausalLMWithValueHead

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


def get_model(model_name, current_device):
    print("Loading subject model...\n")

    kwargs = get_model_loading_kwargs(model_name)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        # load_in_8bit=True,
        device_map={ "": current_device },
        **kwargs,
    )

    print(f"Loaded subject model with {sum(p.numel() for p in model.parameters()):,d} parameters.")
    print(f"Model dtype: {next(iter(model.parameters())).dtype}\n")

    return model