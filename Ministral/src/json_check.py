import json

path = r"D:\AIEngineering\Ministral\src\my_data.jsonl"
with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            if "text" not in data:
                print(f"Line {i}: Missing 'text' key")
            if "[INST]" not in data["text"]:
                print(f"Line {i}: Missing [INST] tag")
        except Exception as e:
            print(f"Line {i}: Invalid JSON - {e}")

print("Check complete!")