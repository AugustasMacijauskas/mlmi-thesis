from tqdm import tqdm
tqdm.pandas()

from collections import Counter

import string
import torch


ALLOWED_ANSWERS = [
    "Yes", "No",
    # "yes", "no",
    # "y", "n", "positive", "negative",
    # "pos", "neg", "1", "0", "true", "false",
    # "correct", "incorrect",
]
CHARACTERS_TO_FILTER = string.punctuation + " \n"


def is_answer_yes_no(answer):
    return answer in ALLOWED_ANSWERS


def postprocess_response(response):
    while response and response[-1] in CHARACTERS_TO_FILTER:
        response = response[:-1]
    return response



def log_samples(ppo_trainer, script_args, epoch, batch_idx, batch):
    # Only print once
    if not ppo_trainer.accelerator.is_main_process: return

    with open(script_args.logging_dir + "text_samples.txt", "a") as f:
        f.write(f"Epoch: {epoch}, batch: {batch_idx}\n\n")

        # Calculate percentage in desired format
        percentage_desired = [int(is_answer_yes_no(x)) for x in batch["response"]]
        percentage_desired = sum(percentage_desired) / len(percentage_desired)
        f.write(f"Percentage desired: {percentage_desired:.4f}\n\n")

        f.write(Counter(batch["response"]).most_common(10).__str__())
        f.write("\n\n")

        for i, (golden_output, output) in enumerate(zip(batch["best_response"], batch["response"])):
            f.write(f"Target: {golden_output}, predicted: {output}\n")

            if i == 7: break

        f.write("-"*100 + "\n\n")


def train(
    ppo_trainer, tokenizer,
    generation_kwargs,
    get_rewards,
    script_args, config,
):
    ppo_trainer.accelerator.print(f"Dataloader length: {len(ppo_trainer.dataloader)}")

    n_epochs = config.steps // len(ppo_trainer.dataloader)
    ppo_trainer.accelerator.print(f"Number of epochs: {n_epochs}")

    for epoch in range(1, n_epochs+1):
        ppo_trainer.accelerator.print(f"Epoch: {epoch}")

        loop = tqdm(
            enumerate(ppo_trainer.dataloader, 1),
            total=len(ppo_trainer.dataloader), leave=False
        )
        for batch_idx, batch in loop:

            question_tensors = batch["input_ids"]

            response_tensors = ppo_trainer.generate(
                question_tensors,
                return_prompt=False,
                batch_size=script_args.generator_batch_size,
                **generation_kwargs,
            )
            responses = tokenizer.batch_decode(
                response_tensors, skip_special_tokens=True,
                spaces_between_special_tokens=False
            )
            if script_args.postprocess_responses:
                responses = [postprocess_response(x) for x in responses]
            batch["response"] = responses

            # Compute rewards (score)
            texts = [q + " " + r for q, r in zip(batch["query"], batch["response"])]
            rewards = get_rewards(texts)

            # Subtract baseline
            rewards -= script_args.reward_baseline

            # Replace reward for undesired answers to -1
            mask = [not is_answer_yes_no(x) for x in batch["response"]]
            # Make the mask a tensor
            mask = torch.tensor(mask, dtype=torch.bool)
            rewards[mask] = -1

            # device_identifier = ppo_trainer.accelerator.local_process_index
            # print(f"DEVICE {device_identifier}: {len(rewards[rewards == -1]) / len(rewards):.4f}")
            
            # Make the rewards a list of tensors
            rewards = [x for x in rewards]

            # Run PPO step
            stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

            # Log samples at the end of batch
            if script_args.log_freq and batch_idx % script_args.log_freq == 0:
                log_samples(ppo_trainer, script_args, epoch, batch_idx, batch)

            # Save at the end of batch
            if script_args.save_freq and batch_idx % script_args.save_freq == 0:
                ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}_{batch_idx}")

        # Save at the end of epoch
        # if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        #     ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
