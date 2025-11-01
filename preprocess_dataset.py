from datasets import load_dataset
from transformers import GPT2TokenizerFast
import collections

label_map = {0: 0, 1: 1, 2: 2, 3: 3}  # integer labels

def load_and_preprocess():
    dataset = load_dataset("ag_news")

    def limit_split(split):
        n = int(len(dataset[split]) * 0.4)
        return dataset[split].shuffle(seed=42).select(range(n))
    
    # Only keep text and integer labels
    def map_labels(example):
        return {
            "text": example["text"],
            "label": label_map[example["label"]]
        }
    
    dataset["train"] = limit_split("train")
    dataset["test"] = limit_split("test")


    dataset = dataset.map(map_labels)

    # Load custom tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer_agnews")
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize text only
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=128
        )

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    train_labels = [int(x) for x in tokenized_dataset["train"]["labels"]]
    print(collections.Counter(train_labels))
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized_dataset, tokenizer
