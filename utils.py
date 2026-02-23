# utils.py

from transformers import DistilBertTokenizerFast
from sklearn.metrics import accuracy_score

MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512


def get_tokenizer():
    return DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


def build_label_maps(labels):
    unique = set(labels)
    label2id = {l: i for i, l in enumerate(unique)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds)}