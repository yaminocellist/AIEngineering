import os
import torch

# 1. CRITICAL: Blackwell (sm_120) & Offline Overrides
os.environ["UNSLOTH_USE_XFORMERS"] = "0"
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["UNSLOTH_USE_TRITON"] = "0" 
os.environ["HF_HUB_OFFLINE"] = "1"

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 2. LOAD MODEL
# Point this exactly to the folder containing your config.json and safetensors
local_model_path = r"D:\Models" 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = local_model_path,
    max_seq_length = 2048,
    dtype = torch.bfloat16, 
    load_in_4bit = True,
    local_files_only = True,
)

# 3. PATCH ATTENTION
if hasattr(model, "config"):
    model.config.use_cache = False
    model.config.attn_implementation = "sdpa" 

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True, 
    random_state = 3407,
)

# 4. DATASET
dataset = load_dataset("csv", data_files={"train": "./dataset/llama_train.csv"})["train"]

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    texts = [alpaca_prompt.format(i, j, k) + tokenizer.eos_token 
             for i, j, k in zip(examples["instruction"], examples["input"], examples["output"])]
    return { "text" : texts }

dataset = dataset.map(formatting_prompts_func, batched = True)

# 5. TRAINING (Fixed Save Strategy)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps = 60,
        learning_rate = 2e-4,
        bf16 = True,
        logging_steps = 1,
        optim = "adamw_torch", 
        output_dir = "outputs",
        save_strategy = "steps", # Change from "no" to "steps"
        save_steps = 50,         # Save a checkpoint every 50 steps
    ),
)

# Force the SDPA kernel
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
    print("\n--- Starting Training (Blackwell Safe Mode) ---")
    trainer.train()

# ==========================================================
# 6. CRITICAL: SAVE THE FINISHED MODEL
# ==========================================================
# We use a raw string for the Windows path
save_path = r"D:\AIEngineering\BERT\llama3_lora_model"

print(f"\n--- Saving LoRA adapters to: {save_path} ---")

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("--- Save Complete! You can now run the test script. ---")