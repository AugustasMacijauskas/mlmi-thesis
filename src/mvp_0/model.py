from peft import LoraConfig

from trl import AutoModelForCausalLMWithValueHead


def get_model(config, current_device):
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