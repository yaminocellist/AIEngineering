# src/train.py
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    AutoTokenizer,
    logging
)
from transformers.trainer_utils import IntervalStrategy
from sklearn.metrics import accuracy_score

from dataset import load_dataset
from model import load_model

# ========================
# Optional: reduce transformers logging noise
# ========================
logging.set_verbosity_info()
logging.enable_default_handler()
logging.enable_explicit_format()


# ========================
# Training Arguments
# ========================
training_args = TrainingArguments(
    output_dir="bert_output",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy=IntervalStrategy.EPOCH,
    save_strategy=IntervalStrategy.EPOCH,
    logging_strategy=IntervalStrategy.EPOCH,
    logging_dir="logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    report_to="none"  # disables wandb or tensorboard reporting
)

# ========================
# Load datasets
# ========================
print("\nLoading datasets...")
train_dataset = load_dataset("dataset/train.csv")
test_dataset = load_dataset("dataset/test.csv")
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}\n")

# ========================
# Load model
# ========================
print("Loading BERT model...")
model = load_model()
print("Model loaded.\n")

# ========================
# Metrics function
# ========================
def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# ========================
# Initialize Trainer
# ========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# ========================
# Train the model
# ========================
print("Starting training...\n")
train_result = trainer.train()

# ========================
# Log final metrics nicely
# ========================
metrics = train_result.metrics
print("\n=== Training Complete ===")
print(f"Train runtime: {metrics['train_runtime']:.2f}s")
print(f"Train samples/sec: {metrics['train_samples_per_second']:.2f}")
print(f"Train steps/sec: {metrics['train_steps_per_second']:.2f}")
print(f"Final train loss: {metrics['train_loss']:.4f}\n")

# Evaluate on test dataset
print("Evaluating on test dataset...\n")
eval_metrics = trainer.evaluate()
print(f"Evaluation loss: {eval_metrics['eval_loss']:.4f}")
print(f"Evaluation accuracy: {eval_metrics['eval_accuracy']:.4f}")
print(f"Eval runtime: {eval_metrics['eval_runtime']:.2f}s")
print(f"Samples/sec: {eval_metrics['eval_samples_per_second']:.2f}")
print(f"Steps/sec: {eval_metrics['eval_steps_per_second']:.2f}\n")

# ========================
# Save the fine-tuned model
# ========================
trainer.save_model("improved_bert")
print("Model saved to 'improved_bert/'")
