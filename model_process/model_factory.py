# models.py
from transformers import GPT2ForSequenceClassification

def get_model(model_name, tokenizer):
    model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=4)
    model.resize_token_embeddings(len(tokenizer))
    return model