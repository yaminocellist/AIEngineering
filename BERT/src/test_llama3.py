import os
import sys
import torch
import warnings
import logging

# 1. THE AGGRESSIVE SILENCER (Must be at the very top)
warnings.filterwarnings("ignore")
# This kills the specific logger causing the 'TypeError' wall of text
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["UNSLOTH_SKIP_STATS"] = "1" 
os.environ["UNSLOTH_USE_XFORMERS"] = "0"
os.environ["XFORMERS_DISABLED"] = "1"

from unsloth import FastLanguageModel

# 2. PATHS
base_model_path = r"D:\Models"
adapter_path = r"D:\AIEngineering\BERT\llama3_lora_model"

# 3. LOAD
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model_path,
    max_seq_length = 2048,
    dtype = torch.bfloat16,
    load_in_4bit = True,
    local_files_only = True,
)
model.load_adapter(adapter_path)
model.to("cuda")
FastLanguageModel.for_inference(model)

# 4. CHAT FUNCTION (Fixed Prompt)
# We use a cleaner prompt to prevent the "Political" or random answers
def chat(user_input):
    prompt = f"### Instruction:\nYou are a helpful assistant. Provide a clear and concise answer.\n\n### Input:\n{user_input}\n\n### Response:\n"
    
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    
    # Updated context manager for Blackwell compatibility
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        outputs = model.generate(
            **inputs, 
            max_new_tokens = 256,
            use_cache = True,
            temperature = 0.7,
            top_p = 0.9,
            do_sample = True,
            pad_token_id = tokenizer.eos_token_id
        )
    
    response = tokenizer.batch_decode(outputs)[0]
    # Split specifically at the last Response tag to get the clean answer
    return response.split("### Response:\n")[1].replace("<|end_of_text|>", "").strip()

# 5. CLEAN LOOP
if __name__ == "__main__":
    # Clear the terminal once at the start to remove any initial import warnings
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("="*50)
    print("   LLAMA-3 ONLINE (RTX 5080)   ")
    print("="*50)

    while True:
        try:
            user_msg = input("\nYou: ")
            if user_msg.lower() in ["exit", "quit"]: break
            
            # Use sys.stdout.write for a cleaner "Thinking..." feel
            sys.stdout.write("Llama: ")
            sys.stdout.flush()
            
            answer = chat(user_msg)
            print(answer)
            print("\n" + "-"*30)
        except KeyboardInterrupt:
            break