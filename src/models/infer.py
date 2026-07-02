from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import LABELS, MAX_LEN

class SentimentClassifier:
    def __init__(self, model_dir):
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()

    def predict(self, texts):
        enc = self.tok(texts, truncation=True, max_length=MAX_LEN,
                       padding=True, return_tensors="pt")
        with torch.no_grad():
            probs = self.model(**enc).logits.softmax(-1)
        out = []
        for text, p in zip(texts, probs):
            label = int(p.argmax())
            out.append({"text": text, "label": label,
                        "label_name": LABELS[label], "score": float(p[label])})
        return out

def split_by_sentiment(results):
    pos = [r["text"] for r in results if r["label"] == 1]
    neg = [r["text"] for r in results if r["label"] == 0]
    return pos, neg
