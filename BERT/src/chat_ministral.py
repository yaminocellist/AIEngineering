import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

# Clear any ghost memory from previous crashes
gc.collect()
torch.cuda.empty_cache()

MODEL_PATH = r"D:\Models\Ministral-8b-Instruct-2410"

print("\n[1/2] Loading Tokenizer and Model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Using float16 to save ~1GB VRAM compared to bfloat16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,   
    device_map="cuda:0", # Forces it specifically onto the GPU, no CPU offloading
    trust_remote_code=True
)

history = [
    {"role": "system", "content": "You are a helpful AI assistant running on an RTX 5080."}
]

def generate_response(user_text):
    global history
    history.append({"role": "user", "content": user_text})
    
    try:
        # Step 1: Get the formatted prompt text
        prompt_text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        
        # Step 2: Manually tokenize to guarantee we get a proper Tensor dictionary
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

        print("...5080 Processing...")
        
        with torch.no_grad():
            # Step 3: Pass inputs['input_ids'] directly - this bypasses the 'shape' error
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Step 4: Decode
        new_tokens = outputs[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        history.append({"role": "assistant", "content": response})
        return response

    except Exception as e:
        history.pop()
        import traceback
        return f"CRASH: {traceback.format_exc()}"

if __name__ == "__main__":
    print(f"\n--- 5080 READY | VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB used ---")
    while True:
        user_input = input("\nUser > ").strip()
        if not user_input: continue
        if user_input.lower() in ["exit", "quit"]: break
        if user_input.lower() == "reset":
            history = [{"role": "system", "content": "You are a helpful AI assistant."}]
            print("Memory cleared.")
            continue
            
        print(f"\nMinistral > {generate_response(user_input)}")