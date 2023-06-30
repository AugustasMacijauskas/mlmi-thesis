from collections import defaultdict

from tqdm import tqdm
tqdm.pandas()

import torch

from datasets import Dataset


def preprocess_function(text, template, tokenization_function):
    entry = { "text": text }

    # Get the positive and negative examples
    entry["label"] = 1
    pos_q, pos_a = template.apply(entry)

    entry["label"] = 0
    neg_q, neg_a = template.apply(entry)

    # Tokenize the inputs
    pos_inputs = tokenization_function(pos_q, pos_a)
    neg_inputs = tokenization_function(neg_q, neg_a)

    output = {
        "pos_input_ids": pos_inputs["input_ids"].squeeze(),
        "pos_attention_mask": pos_inputs["attention_mask"].squeeze(),
        "pos_labels": pos_inputs["labels"].squeeze(),
        "neg_input_ids": neg_inputs["input_ids"].squeeze(),
        "neg_attention_mask": neg_inputs["attention_mask"].squeeze(),
        "neg_labels": neg_inputs["labels"].squeeze(),
    }

    return output


def pad(features, tokenizer, device):
    return tokenizer.pad(
        features, padding="longest", return_tensors="pt",
    ).to(device)

def get_rewards(texts, reward_model, reward_model_tokenizer, template, device):
    text_dataset = Dataset.from_dict({ "text": texts })

    def tokenization_function(q, a):
        return reward_model_tokenizer(
            q, text_target=a.strip(),
            add_special_tokens=True, return_tensors="pt",
        )

    processed_dataset = text_dataset.map(
        preprocess_function, batched=False,
        remove_columns=text_dataset.column_names,
        fn_kwargs={ "template": template, "tokenization_function": tokenization_function },
    )

    # Get max length
    max_length = max([len(item) for item in processed_dataset["pos_input_ids"]])

    # Pad pos features
    features_to_pad = {
        "input_ids": processed_dataset["pos_input_ids"],
        "attention_mask": processed_dataset["pos_attention_mask"],
    }
    pos_inputs = pad(features_to_pad, reward_model_tokenizer, device)
    assert all([item.shape[1] == max_length for item in pos_inputs.values()])
    pos_inputs["labels"] = torch.LongTensor(processed_dataset["pos_labels"]).to(device)

    # Pad neg features
    features_to_pad = {
        "input_ids": processed_dataset["neg_input_ids"],
        "attention_mask": processed_dataset["neg_attention_mask"],
    }
    neg_inputs = pad(features_to_pad, reward_model_tokenizer, device)
    assert all([item.shape[1] == max_length for item in neg_inputs.values()])
    neg_inputs["labels"] = torch.LongTensor(processed_dataset["neg_labels"]).to(device)

    # Run model
    with torch.no_grad():
        predictions = reward_model(pos_inputs, neg_inputs)

    return predictions.sigmoid()


def train(
    ppo_trainer, tokenizer,
    generation_kwargs,
    reward_model, reward_model_tokenizer,
    template,
    script_args, config,
    device,
):
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        # TODO: what is this for? -> Probably handles gradient accumulation
        if epoch >= config.total_ppo_epochs:
            break

        question_tensors = batch["input_ids"]

        max_new_tokens = max(batch["response_len"])
        max_new_tokens = max(4, max_new_tokens)
        generation_kwargs["max_new_tokens"] = max_new_tokens

        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            # length_sampler=output_length_sampler, # TODO: can be none
            batch_size=4, # TODO: generations are made in batches
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["prompt"], batch["response"])]
        rewards = get_rewards(texts, reward_model, reward_model_tokenizer, template, device)
        rewards -= script_args.reward_baseline
        rewards = [x for x in rewards]

        # Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
