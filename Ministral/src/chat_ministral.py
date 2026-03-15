import os
import torch
from unsloth import FastLanguageModel

# 1. Total Offline Mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*_check_is_size will be removed.*")

# 2. Setup
model_path = r"D:\Models\Ministral-8b-Instruct-2410"
lora_path = "ministral_customer_service_lora"

print("Loading model onto Blackwell RTX 5080...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 4096,
    load_in_4bit = True,
    local_files_only = True,
)

model = FastLanguageModel.for_inference(model)
model.load_adapter(lora_path)
model.to("cuda")

print("\n" + "="*50)
print("MINISTRAL CUSTOMER SERVICE BOT ACTIVE")
print("Type 'exit' or 'quit' to stop.")
print("="*50 + "\n")

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    messages = [{"role": "user", "content": user_input}]
    
    # Process inputs
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    # Generate
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 256,
        use_cache = True,
        pad_token_id = tokenizer.eos_token_id # Prevents the attention_mask warning
    )

    response = tokenizer.batch_decode(outputs)
    
    # Display logic
    full_text = response[0]
    if "assistant" in full_text:
        answer = full_text.split("assistant")[-1].strip()
    else:
        answer = full_text.split("[/INST]")[-1].strip()
    
    # Strip the final EOS token for a clean UI
    answer = answer.replace("</s>", "").replace("<|end_of_text|>", "").strip()
    
    print(f"\nBot: {answer}\n" + "-"*30)