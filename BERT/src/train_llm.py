import torch
import pandas as pd
import json

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)

MODEL_NAME = "gpt2"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)

df = pd.read_csv("dataset/train.csv")

def format_chat(row):

    conversation = json.loads(row["conversation"])

    text = ""
    for msg in conversation:
        text += f"{msg['role']}: {msg['content']}\n"

    return text

df["text"] = df.apply(format_chat, axis=1)

dataset = Dataset.from_pandas(df)

def tokenize(batch):

    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

dataset = dataset.map(tokenize, batched=True)

dataset.set_format(
    type="torch",
    columns=["input_ids","attention_mask"]
)

training_args = TrainingArguments(

    output_dir="llm_output",

    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    num_train_epochs=3,

    logging_steps=50,
    save_steps=1000,

    fp16=True,

    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

trainer.save_model("my_llm")
