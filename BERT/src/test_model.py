from transformers import pipeline

# Load your trained model and tokenizer from the saved folder
classifier = pipeline("text-classification", model="./improved_llm")

# Test cases - notice these aren't exactly what's in your generator!
test_sentences = [
    "Translate English to French: 'The cat is on the table.'",
    "Classify the sentiment: 'I absolutely love how this model is working!'",
    "Answer the question: 'What is the capital of France?'",
    "Extract keywords: 'machine learning, neurons, weights, backpropagation'",
    "Summarize the following text: 'The weather was cold but the sun was shining brightly over the mountains.'"
]

print("\n--- Model Predictions ---")
for sentence in test_sentences:
    result = classifier(sentence)
    label = result[0]['label']
    score = result[0]['score']
    print(f"Text: {sentence}")
    print(f"Prediction: {label} (Confidence: {score:.4f})\n")