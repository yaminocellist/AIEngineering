import csv
import random
from faker import Faker
import os

fake = Faker()
Faker.seed(42)
random.seed(42)

os.makedirs("./dataset", exist_ok=True)

def generate_llama_csv(filename, count):
    print(f"Generating {count} generative rows for Llama-3...")
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "input", "output"])
        writer.writeheader()
        
        for _ in range(count):
            task_choice = random.random()
            
            # TASK 1: Translation
            if task_choice < 0.2:
                text = fake.sentence()
                instruction = "Translate the following sentence from English to French."
                input_data = text
                # Note: In a real scenario, you'd use a translation API here.
                # For this demo, we use a placeholder "fake French" (reversed text).
                output_data = f"French translation: {text[::-1]}"
            
            # TASK 2: Summarization
            elif task_choice < 0.4:
                text = " ".join(fake.sentences(nb=4))
                instruction = "Summarize the provided text into a single short sentence."
                input_data = text
                output_data = fake.sentence(nb_words=6)
            
            # TASK 3: Sentiment Analysis
            elif task_choice < 0.6:
                sentiment = random.choice(["Positive", "Negative", "Neutral"])
                instruction = "Identify the sentiment of the provided text."
                input_data = fake.sentence()
                output_data = f"The sentiment of the text is {sentiment}."
            
            # TASK 4: General QA
            elif task_choice < 0.8:
                instruction = "Provide a concise answer to the following question."
                input_data = fake.sentence().rstrip(".") + "?"
                output_data = fake.word().capitalize() + " is the primary answer."
                
            # TASK 5: Keyword Extraction
            else:
                instruction = "List the most important keywords from the text below."
                input_data = " ".join(fake.words(nb=12))
                output_data = ", ".join(fake.words(nb=4))

            writer.writerow({
                "instruction": instruction,
                "input": input_data,
                "output": output_data
            })

if __name__ == "__main__":
    # Generating 10k rows. With an RTX 5080, Llama-3 will eat through this in minutes.
    generate_llama_csv("./dataset/llama_train.csv", 10000)
    print("DONE: Dataset created at ./dataset/llama_train.csv")