from transformers import AutoModelForSequenceClassification

MODEL_NAME = "bert-base-uncased"

def load_model():

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    return model
