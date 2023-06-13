import torch

from transformers import pipeline


# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
def get_reward_model(script_args, ppo_trainer, tokenizer, current_device):
    reward_model_name = script_args.reward_model_name

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
    
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=reward_model_name,
        device_map={ "": current_device },
        model_kwargs={ "load_in_8bit": True },
        tokenizer=tokenizer,
        return_token_type_ids=False,
    )

    return sentiment_pipe
