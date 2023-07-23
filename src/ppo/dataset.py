from datasets import load_dataset, DatasetDict

from accelerate import Accelerator
temporary_accelerator = Accelerator()


def preprocess_dataset(
    dataset,
    tokenizer,
    num_proc=12,
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


def get_dataset(dataset_name, tokenizer, num_proc=12, subsets_to_delete=None):
    print("Loading dataset...\n")

    dataset = load_dataset(dataset_name, split="train")
    original_column_names = dataset.column_names # will be removed later
    original_column_names.remove("prompt") # but want to keep the prompt
    original_column_names.remove("best_response") # and want to keep the prompt

    # Delete certain subsets if desired
    if subsets_to_delete is not None:
        for subset in subsets_to_delete:
            print(f"Deleting subset: {subset}")
            dataset = dataset.filter(lambda x: x["original_dataset"] != subset)

    # Get max prompt length
    prompt_max_len = max(
        tokenizer(row["prompt"], return_tensors="pt")["input_ids"].shape[1] for row in dataset
    )
    print(f"\nMax prompt length: {prompt_max_len}\n")

    # Do not need to truncate for GPT-J 6B or dolly-v2
    # check for other models
    processed_dataset = dataset.map(
        lambda batch: tokenizer(batch["prompt"]), batched=True, num_proc=num_proc
    )

    # Add max response length
    def add_max_response_len(row):
        row["response_len"] = tokenizer(row["best_response"], return_tensors="pt")["input_ids"].shape[1]
        return row

    processed_dataset = processed_dataset.map(
        add_max_response_len, batched=False, num_proc=num_proc
    )

    response_max_len = max(processed_dataset["response_len"])
    print(f"Max response length: {response_max_len}\n")

    # Remove original columns
    processed_dataset = processed_dataset.remove_columns(original_column_names)
    print(f"Remaining columns: {processed_dataset.column_names}\n")

    # Shuffle and sample first n examples
    processed_dataset = processed_dataset.shuffle(seed=42).select(range(8192))

    # Set format
    processed_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True)

    print(f"Total number of examples: {len(processed_dataset)}\n")
    
    print("Processing finished.\n")

    return processed_dataset, prompt_max_len, response_max_len


def get_dataset_qnli(dataset_name, tokenizer, num_proc=12, margin=8, num_examples=8192, seed=42):
    temporary_accelerator.print("Loading dataset...\n")

    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.rename_column("prompt", "query")
    original_column_names = dataset.column_names # will be removed later
    original_column_names.remove("query") # but want to keep the prompt
    original_column_names.remove("best_response") # and want to keep the prompt

    # Do not need to truncate for GPT-J 6B or dolly-v2
    # check for other models
    processed_dataset = dataset.map(
        lambda batch: tokenizer(batch["query"]), batched=True, num_proc=num_proc
    )

    # Filter too long examples
    # Substract a bit more to allow for generations to be processed
    filter_too_long = lambda batch: [
        len(x) < tokenizer.model_max_length - margin for x in batch["input_ids"]
    ]
    processed_dataset = processed_dataset.filter(
        filter_too_long, batched=True, num_proc=12
    )

    # Remove original columns
    processed_dataset = processed_dataset.remove_columns(
        original_column_names + ["token_type_ids"]
    )
    temporary_accelerator.print(f"Remaining columns: {processed_dataset.column_names}\n")

    # Shuffle and sample first n examples
    processed_dataset = processed_dataset.shuffle(seed=seed).select(range(num_examples))

    # Set format
    processed_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask"],
        output_all_columns=True
    )

    temporary_accelerator.print(f"Total number of examples: {len(processed_dataset)}\n")
    
    temporary_accelerator.print("Data preprocessing finished.\n")

    return processed_dataset


def get_dataset_trlx(dataset_name, eval_dataset_size, train_dataset_size=8192, subsets_to_delete=None):
    print("Loading dataset...\n")

    dataset = load_dataset(dataset_name, split="train")

    # Delete certain subsets if desired
    if subsets_to_delete is not None:
        for subset in subsets_to_delete:
            print(f"Deleting subset: {subset}")
            dataset = dataset.filter(lambda x: x["original_dataset"] != subset)

    # Remove redundant columns
    dataset = dataset.rename_column("best_response", "original_output")
    
    original_column_names = dataset.column_names # will be removed later
    original_column_names.remove("prompt") # but want to keep the prompt
    original_column_names.remove("original_output") # and want to keep the prompt

    dataset = dataset.remove_columns(original_column_names)
    print(f"Remaining columns: {dataset.column_names}\n")

    # Shuffle and sample first n examples
    train_dataset = dataset.shuffle(seed=42).select(range(train_dataset_size))
    eval_dataset = dataset.shuffle(seed=42).select(range(train_dataset_size, train_dataset_size + eval_dataset_size))

    processed_dataset = DatasetDict({"train": train_dataset, "eval": eval_dataset})

    # Set format

    print(f"Total number of train examples: {len(processed_dataset['train'])}\n")
    print(f"Total number of eval examples: {len(processed_dataset['eval'])}\n")
    
    print("Processing finished.\n")

    return processed_dataset


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
