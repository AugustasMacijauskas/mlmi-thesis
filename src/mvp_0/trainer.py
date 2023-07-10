from collections import defaultdict

from tqdm import tqdm
tqdm.pandas()

import torch

from datasets import Dataset


def train(
    ppo_trainer, tokenizer,
    generation_kwargs,
    get_rewards,
    script_args, config,
):
    print(f"Dataloader length: {len(ppo_trainer.dataloader)}")

    n_epochs = config.steps // len(ppo_trainer.dataloader)
    print(f"Number of epochs: {n_epochs}")

    for epoch in range(1, n_epochs+1):
        print(f"Epoch: {epoch}")

        loop = tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader), leave=False)
        for _, batch in loop:

            question_tensors = batch["input_ids"]

            # Set generation_kwargs
            max_new_tokens = generation_kwargs.get(
                "max_new_tokens", max(batch["response_len"])
            )
            max_new_tokens = max(4, max_new_tokens)
            generation_kwargs["max_new_tokens"] = max_new_tokens

            response_tensors = ppo_trainer.generate(
                question_tensors,
                return_prompt=False,
                batch_size=script_args.generator_batch_size,
                **generation_kwargs,
            )
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            # Compute sentiment score
            texts = [q + r for q, r in zip(batch["prompt"], batch["response"])]
            rewards = get_rewards(texts)
            rewards -= script_args.reward_baseline
            rewards = [x for x in rewards]

            # Run PPO step
            stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

        # Save at the end of epoch
        if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
