import csv
import random
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

# Define task types and labels
tasks = [
    ("translate", 0),
    ("summarize", 1),
    ("sentiment", 2),
    ("qa", 3),
    ("keywords", 4),
]

# Number of examples
num_examples = 100_000

# CSV file
output_file = "./dataset/test_llm.csv"

def generate_prompt_completion(task):
    task_type, label = task
    if task_type == "translate":
        text = fake.sentence(nb_words=random.randint(5, 12))
        completion = f"French: {text[::-1]}"  # fake French-like output
        prompt = f"Translate English to French: '{text}'"
    elif task_type == "summarize":
        text = " ".join(fake.sentences(nb=random.randint(2, 5)))
        completion = " ".join(fake.sentences(nb=1))
        prompt = f"Summarize the following text: '{text}'"
    elif task_type == "sentiment":
        sentiment = random.choice(["Positive", "Negative", "Neutral"])
        text = fake.sentence()
        completion = sentiment
        prompt = f"Classify the sentiment: '{text}'"
    elif task_type == "qa":
        question = fake.sentence(nb_words=random.randint(4, 8)).rstrip("?") + "?"
        answer = fake.word()
        prompt = f"Answer the question: '{question}'"
        completion = answer
    elif task_type == "keywords":
        text = " ".join(fake.words(nb=random.randint(5, 12)))
        completion = ", ".join(fake.words(nb=random.randint(2, 5)))
        prompt = f"Extract keywords: '{text}'"
    return prompt, completion, label

with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["prompt", "completion", "label"])
    
    for _ in range(num_examples):
        task = random.choice(tasks)
        prompt, completion, label = generate_prompt_completion(task)
        writer.writerow([prompt, completion, label])

print(f"Test dataset generated: {output_file} with {num_examples} examples")
