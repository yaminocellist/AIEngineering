import csv
import random
from faker import Faker
import os

fake = Faker()
Faker.seed(42)
random.seed(42)

os.makedirs("./dataset", exist_ok=True)

TASKS = [
    {"type": "translate", "label": 0, "template": "Translate English to French: '{text}'"},
    {"type": "summarize", "label": 1, "template": "Summarize the following text: '{text}'"},
    {"type": "sentiment", "label": 2, "template": "Classify the sentiment: '{text}'"},
    {"type": "qa", "label": 3, "template": "Answer the question: '{text}'"},
    {"type": "keywords", "label": 4, "template": "Extract keywords: '{text}'"},
]

def get_completion(task_type):
    if task_type == "translate":
        return f"French: {fake.sentence()[::-1]}"
    if task_type == "summarize":
        return fake.sentence(nb_words=6)
    if task_type == "sentiment":
        return random.choice(["Positive", "Negative", "Neutral"])
    if task_type == "qa":
        return fake.word().capitalize()
    if task_type == "keywords":
        return ", ".join(fake.words(nb=3))
    return "N/A"

def create_csv(filename, count):
    print(f"Creating {filename}...")
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "completion", "label"])
        writer.writeheader()

        for _ in range(count):
            task = random.choice(TASKS)

            if task["type"] == "qa":
                text = fake.sentence().rstrip(".") + "?"
            else:
                text = fake.sentence()

            writer.writerow({
                "prompt": task["template"].format(text=text),
                "completion": get_completion(task["type"]),
                "label": task["label"]
            })

if __name__ == "__main__":
    create_csv("./dataset/train_llm.csv", 100000)
    create_csv("./dataset/test_llm.csv", 5000)
    print("DONE: CSV files created in ./dataset/ folder.")
