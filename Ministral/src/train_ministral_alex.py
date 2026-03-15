import os
import torch
import warnings
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Environment & Blackwell Fixes
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["UNSLOTH_OFFLINE"] = "1" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

# 2. Configuration
model_path = r"D:\Models\Ministral-8b-Instruct-2410"
data_path = r"D:\AIEngineering\Ministral\src\alex_store_manager_dataset.jsonl"
output_dir = "ministral_customer_service_lora"
max_seq_length = 1024 

# 3. Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    local_files_only = True,
)

# --- 4. OPTIMIZED BLACKWELL PATCH ---
import unsloth_zoo.fused_losses.cross_entropy_loss as ce_loss
def fixed_get_chunk_size(*args, **kwargs):
    # 1024 is safe, but for a 5080 we want to avoid overhead
    return 1024 
ce_loss.get_chunk_size = fixed_get_chunk_size
print(">>> Blackwell High-Speed Patch (1024) Applied.")

# 5. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    # SPEED OPTIMIZATION: Disable checkpointing since 5080 has enough VRAM
    use_gradient_checkpointing = False, 
    random_state = 3407,
)

# 6. Load Dataset
dataset = load_dataset("json", data_files={"train": data_path}, split="train")

# 7. Trainer Setup (Turbo Settings)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = True, 
    args = TrainingArguments(
        # SPEED OPTIMIZATION: Increase batch size to saturate the 5080
        per_device_train_batch_size = 8, 
        gradient_accumulation_steps = 1, # Direct updates are faster
        warmup_steps = 5,
        max_steps = 100, # With packing and larger batch, 100 steps covers more data
        learning_rate = 2e-4,
        fp16 = False,   
        bf16 = True, 
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
print("Starting Turbo Training on RTX 5080...")
trainer.train()

# 9. Save
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Success! Model saved to {output_dir}")