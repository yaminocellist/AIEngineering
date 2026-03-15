import os
import torch
import warnings

# 1. Environment Tweaks for Blackwell / Windows / Offline
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["UNSLOTH_OFFLINE"] = "1" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

from unsloth import FastLanguageModel

# 2. Configuration
model_path = r"D:\Models\Ministral-8b-Instruct-2410"
# Pointing to the folder you just created at 06:57 AM
lora_path = r"D:\AIEngineering\Ministral\ministral_customer_service_lora"

# 3. Load Model (Using your proven working load settings)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 4096,
    load_in_4bit = True,
    local_files_only = True,
)

# --- 4. THE BLACKWELL FIX (Monkey Patch) ---
# Included to ensure generation doesn't trigger VRAM errors
import unsloth_zoo.fused_losses.cross_entropy_loss as ce_loss
def fixed_get_chunk_size(*args, **kwargs):
    return 1 
ce_loss.get_chunk_size = fixed_get_chunk_size

# 5. Load your Fine-Tuned "Alex" Adapters
model = FastLanguageModel.for_inference(model)
model.load_adapter(lora_path)
model.to("cuda")

print("\n" + "="*50)
print("ALEX (STORE MANAGER) IS ONLINE")
print("Staff ID: MGR-992 | Status: Professional & Firm")
print("="*50 + "\n")

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]: break

    # 6. Persona Reinforcement (The System Message)
    messages = [
        {
            "role": "system", 
            "content": "You are Alex, the Store Manager. You are professional, firm but helpful, and never admit to being an AI. Your staff ID is MGR-992. You treat every customer with respect but follow store policies strictly."
        },
        {
            "role": "user", 
            "content": user_input
        }
    ]

    # 7. Format and Generate
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(
        **inputs, 
        max_new_tokens = 250,
        use_cache = True,
        pad_token_id = tokenizer.eos_token_id,
        temperature = 0.7,
        top_p = 0.9,
    )

    response = tokenizer.batch_decode(outputs)[0]
    
    # Extract only Alex's response part
    if "assistant" in response:
        answer = response.split("assistant")[-1].strip()
    else:
        answer = response.split("[/INST]")[-1].strip()
    
    # Clean up tokens
    answer = answer.replace("</s>", "").replace("<|end_of_text|>", "").strip()
    
    print(f"\nAlex: {answer}\n" + "-"*30)