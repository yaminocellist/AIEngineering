import os
import torch

# 1. Environment Tweaks for Blackwell / Windows
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["UNSLOTH_OFFLINE"] = "1" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 2. Configuration
model_path = r"D:\Models\Ministral-8b-Instruct-2410"
data_path = r"D:\AIEngineering\Ministral\src\my_data.jsonl"
output_dir = "ministral_customer_service_lora"

# 3. Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 4096,
    load_in_4bit = True,
)

# --- 4. THE BLACKWELL FIX (Monkey Patch) ---
# This reaches into the Unsloth library and replaces the bugged 
# VRAM check with the standard, working PyTorch loss.
import unsloth_zoo.fused_losses.cross_entropy_loss as ce_loss

def fixed_get_chunk_size(*args, **kwargs):
    # This force-returns a chunk size of 1, bypassing the VRAM check
    return 1 

ce_loss.get_chunk_size = fixed_get_chunk_size
print("Blackwell VRAM Patch Applied.")

# 5. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 6. Load Dataset
dataset = load_dataset("json", data_files={"train": data_path}, split="train")

# 7. Trainer Setup
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 4096,
    args = TrainingArguments(
        per_device_train_batch_size = 1, 
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = False,   
        bf16 = True,    # RTX 5080 must use BF16
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
        report_to = "none",
    ),
)

# 8. Start Training
print("Starting training on RTX 5080...")
trainer.train()

# 9. Save
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Success! Model saved to {output_dir}")