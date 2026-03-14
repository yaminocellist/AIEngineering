import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

id2label = {0: "translate", 1: "summarize", 2: "sentiment", 3: "qa", 4: "keywords"}
label2id = {v: k for k, v in id2label.items()}

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5,
    id2label=id2label,
    label2id=label2id
)

dataset = load_dataset(
    "csv",
    data_files={
        "train": "./dataset/train_llm.csv",
        "test": "./dataset/test_llm.csv"
    }
)

def tokenize_function(batch):
    # Use str(p) and str(c) to handle any missing (None) values safely
    texts = [str(p) + " " + str(c) for p, c in zip(batch["prompt"], batch["completion"])]
    tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=128)
    tokenized["labels"] = batch["label"]
    return tokenized

tokenized_datasets = dataset.map(tokenize_function, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

training_args = TrainingArguments(
    output_dir="llm_output",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    report_to="none",
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,  # ✅ New keyword
    compute_metrics=compute_metrics
)

print("Starting training loop...")
trainer.train()

trainer.save_model("improved_llm")

print("DONE: Training finished. Model saved to ./improved_llm")
