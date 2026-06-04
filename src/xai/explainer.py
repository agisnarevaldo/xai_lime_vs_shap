import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class PredictProbaWrapper:
    """
    Wrapper class that binds tokenizer, model, and device to produce a callable
    predict_proba function compatible with LIME and SHAP.
    """
    def __init__(self, model, tokenizer, device, max_len=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.model.eval().to(self.device)

    def __call__(self, texts):
        # Normalize inputs to list of strings
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            texts = list(texts)
            
        texts = [str(t) if t is not None else "" for t in texts]
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        # Move inputs to target device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
        return probs

def load_sentiment_model_and_tokenizer(model_dir, tokenizer_dir, device="cpu"):
    """
    Utility function to load the model and tokenizer from specified directories.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval().to(device)
    return model, tokenizer
