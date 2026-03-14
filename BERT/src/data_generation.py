import csv
import random
import textwrap

OUTPUT_FILE = "dataset/train.csv"
NUM_SAMPLES = 50000

languages = [
    ("French","Bonjour le monde"),
    ("Spanish","Hola mundo"),
    ("German","Hallo Welt"),
    ("Italian","Ciao mondo"),
]

topics = [
    "neural networks",
    "transformers",
    "gradient descent",
    "reinforcement learning",
    "GPU computing",
    "distributed systems",
    "parallel programming",
    "deep learning",
]

dialogue_questions = [
    "How can I start learning machine learning?",
    "What programming language is best for AI?",
    "How does a GPU accelerate deep learning?",
    "What is the difference between AI and machine learning?"
]

coding_tasks = [
    ("reverse a string",
     "def reverse_string(s):\n    return s[::-1]"),

    ("compute fibonacci numbers",
     textwrap.dedent("""\
     def fibonacci(n):
         a,b = 0,1
         seq = []
         for _ in range(n):
             seq.append(a)
             a,b = b,a+b
         return seq
     """)),

    ("check if a number is prime",
     textwrap.dedent("""\
     def is_prime(n):
         if n<=1:
             return False
         for i in range(2,int(n**0.5)+1):
             if n % i == 0:
                 return False
         return True
     """))
]

summaries = [
    (
        "Deep learning models learn hierarchical representations from data using multilayer neural networks.",
        "Deep learning trains multilayer neural networks to learn hierarchical features."
    ),
    (
        "Transformers rely on self-attention mechanisms to process sequences efficiently.",
        "Transformers use attention mechanisms instead of recurrence."
    )
]

math_problems = [
    ("If a train travels 60 km in 1 hour, how far will it travel in 3 hours?",
     "The train travels 60 km per hour. In 3 hours it travels 180 km.",
     "180 km"),

    ("A rectangle has width 5 and height 10. What is its area?",
     "Area = width × height = 5 × 10 = 50",
     "50"),

    ("John has 10 apples and gives away 4. How many remain?",
     "10 − 4 = 6 apples remain.",
     "6")
]

knowledge_questions = [
    ("What is CUDA?",
     "CUDA is a GPU computing platform developed by NVIDIA that enables parallel programming on GPUs."),

    ("What is overfitting in machine learning?",
     "Overfitting occurs when a model learns the training data too well and fails to generalize."),

    ("What is gradient descent?",
     "Gradient descent is an optimization algorithm used to minimize loss functions in machine learning.")
]

reasoning_problems = [
    ("Alice is older than Bob. Bob is older than Charlie. Who is the oldest?",
     "Alice > Bob > Charlie therefore Alice is the oldest.",
     "Alice"),

    ("All cats are animals. Some animals are black. Can we conclude all cats are black?",
     "No, because only some animals are black. Cats might not be among them.",
     "No")
]


# ------------------------------------------------
# Task generators
# ------------------------------------------------

def gen_explanation():
    topic = random.choice(topics)
    instruction = f"Explain the concept of {topic} in machine learning."
    output = f"{topic.capitalize()} is an important concept in modern AI systems and is widely used in machine learning research."
    return instruction, "", output


def gen_translation():
    lang, translation = random.choice(languages)
    instruction = f"Translate the sentence into {lang}."
    input_text = "Hello world"
    output = translation
    return instruction, input_text, output


def gen_summary():
    text, summary = random.choice(summaries)
    return "Summarize the following text.", text, summary


def gen_code():
    task, solution = random.choice(coding_tasks)
    instruction = f"Write Python code to {task}."
    return instruction, "", solution


def gen_debug():
    instruction = "Fix the bug in the following Python code."
    input_text = "def add(a,b)\n return a+b"
    output = textwrap.dedent("""\
    def add(a,b):
        return a+b
    """)
    return instruction, input_text, output


def gen_math():
    problem, reasoning, answer = random.choice(math_problems)
    instruction = "Solve the math problem step by step."
    output = reasoning + "\nFinal answer: " + answer
    return instruction, problem, output


def gen_knowledge():
    q, a = random.choice(knowledge_questions)
    return q, "", a


def gen_reasoning():
    problem, reasoning, answer = random.choice(reasoning_problems)
    instruction = "Answer the reasoning question."
    output = reasoning + "\nAnswer: " + answer
    return instruction, problem, output


def gen_dialogue():
    question = random.choice(dialogue_questions)
    answer = "That is a great question. A good starting point is learning Python and basic machine learning concepts."
    return question, "", answer


generators = [
    gen_explanation,
    gen_translation,
    gen_summary,
    gen_code,
    gen_debug,
    gen_math,
    gen_knowledge,
    gen_reasoning,
    gen_dialogue
]


# ------------------------------------------------
# Dataset generation
# ------------------------------------------------

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["instruction","input","output"])

    for _ in range(NUM_SAMPLES):
        g = random.choice(generators)
        inst, inp, out = g()
        writer.writerow([inst, inp, out])

print(f"Generated {NUM_SAMPLES} samples -> {OUTPUT_FILE}")
