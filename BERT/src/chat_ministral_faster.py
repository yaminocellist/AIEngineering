import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_PATH = r"D:\Models\Ministral-8b-Instruct-2410"

print(f"\n[RTX 5080] Loading Model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16, # Using bfloat16 for Blackwell native support
    device_map="cuda",
    trust_remote_code=True,
    attn_implementation="sdpa"
)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
history = []

def chat():
    global history
    print("\n--- 5080 READY ---")
    
    while True:
        user_input = input("\nYou > ").strip()
        if not user_input: continue
        if user_input.lower() in ["exit", "quit"]: break
        
        history.append({"role": "user", "content": user_input})
        
        # 1. Get the encoded output as a dictionary
        # We don't use 'return_tensors' here to avoid the BatchEncoding container bug
        encoded_dict = tokenizer.apply_chat_template(
            history, 
            add_generation_prompt=True, 
            return_dict=True, 
            return_tensors="pt"
        ).to("cuda")

        # 2. MANUALLY extract the Tensors
        # This bypasses the 'AttributeError' because we aren't passing a container
        input_ids = encoded_dict["input_ids"]
        attention_mask = encoded_dict["attention_mask"]

        print("\nMinistral > ", end="")
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                streamer=streamer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # 3. Save response to history
            # We slice the output to remove the prompt tokens
            new_tokens = output_ids[0][input_ids.shape[-1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    chat()