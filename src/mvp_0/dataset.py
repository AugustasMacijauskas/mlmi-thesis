from datasets import load_dataset


def preprocess_dataset(
    dataset,
    tokenizer,
    num_proc=24,
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names, # remove the original columns
    )
    dataset = dataset.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    dataset.set_format(type="torch")
    
    return dataset


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def get_dataset(dataset_name, tokenizer, num_proc=12):
    print("Loading dataset...\n")

    dataset = load_dataset("AugustasM/burns-ppo-training-dataset", split="train")

    subsets_to_delete = ["piqa"]
    for subset in subsets_to_delete:
        print(f"Deleting subset: {subset}")
        dataset = dataset.filter(lambda x: x["original_dataset"] != subset)

    # Do not need to truncate for GPT-J 6B, check for other models
    def tokenize(batch, max_length=1024):
        return tokenizer(
            batch["prompt"], padding="max_length",
            max_length=max_length, return_tensors="pt",
        )

    prompt_max_len = max(
        tokenizer(row["prompt"], return_tensors="pt")["input_ids"].shape[1] for row in dataset
    )
    print(f"\nMax prompt length: {prompt_max_len}\n")

    processed_dataset = dataset.map(
        tokenize, batched=True, num_proc=num_proc,
        fn_kwargs={ "max_length": prompt_max_len }
    )
    processed_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Assert that all rows in the dataset have the same length
    assert len(set(row["input_ids"].shape[0] for row in processed_dataset)) == 1

    response_max_len = max(
        tokenizer(row["best_response"], return_tensors="pt")["input_ids"].shape[1] for row in dataset
    )
    print(f"Max response length: {response_max_len}")

    return processed_dataset, prompt_max_len, response_max_len


def main():
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"{tokenizer.model_max_length=}")
    dataset, _, _ = get_dataset(
        "AugustasM/burns-ppo-training-dataset", tokenizer,
    )
    print(dataset)


if __name__ == "__main__":
    main()
