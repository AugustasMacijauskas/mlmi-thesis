from tqdm import tqdm
tqdm.pandas()


def log_samples(ppo_trainer, script_args, epoch, batch_idx, batch):
    # Only print once
    if not ppo_trainer.accelerator.is_main_process: return

    with open(script_args.logging_dir + "text_samples.txt", "a") as f:
        f.write(f"Epoch: {epoch}, batch: {batch_idx}\n\n")
        for i, (golden_output, output) in enumerate(zip(batch["golden_output"], batch["response"])):
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
            enumerate(ppo_trainer.dataloader),
            total=len(ppo_trainer.dataloader), leave=False
        )
        for batch_idx, batch in loop:

            question_tensors = batch["input_ids"]

            # Set generation_kwargs
            # max_new_tokens = generation_kwargs.get(
            #     "max_new_tokens", max(batch["response_len"])
            # )
            # max_new_tokens = max(4, max_new_tokens)
            # generation_kwargs["max_new_tokens"] = max_new_tokens

            response_tensors = ppo_trainer.generate(
                question_tensors,
                return_prompt=False,
                batch_size=script_args.generator_batch_size,
                **generation_kwargs,
            )
            batch["response"] = tokenizer.batch_decode(
                response_tensors, skip_special_tokens=True,
                spaces_between_special_tokens=False
            )

            # Compute rewards (score)
            texts = [q + " " + r for q, r in zip(batch["query"], batch["response"])]
            # if batch_idx == 0:
            #     Accelerator().print(f"Example text: {texts[0]}")
            
            rewards = get_rewards(texts)

            # Subtract baseline
            rewards -= script_args.reward_baseline
            rewards = [x for x in rewards]

            # Run PPO step
            stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
        
            # if script_args.log_freq and batch_idx and batch_idx % script_args.log_freq == 0:
            #     log_samples(ppo_trainer, script_args, epoch, batch_idx, batch)

        # Save at the end of epoch
        if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
