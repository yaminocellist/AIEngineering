import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_dataset(path):

    df = pd.read_csv(path)

    dataset = Dataset.from_pandas(df)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    dataset = dataset.map(tokenize)

    dataset = dataset.rename_column("label", "labels")

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return dataset
