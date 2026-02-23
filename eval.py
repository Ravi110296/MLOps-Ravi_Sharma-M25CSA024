# eval.py

import random
from collections import defaultdict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from transformers import DistilBertForSequenceClassification, Trainer

from data import get_data, MyDataset
from utils import get_tokenizer, build_label_maps


def main():

    # Rebuild test data exactly as during training
    train_texts, train_labels, test_texts, test_labels = get_data()

    label2id, id2label = build_label_maps(train_labels)

    tokenizer = get_tokenizer()

    test_enc = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
    test_labels_enc = [label2id[y] for y in test_labels]

    test_dataset = MyDataset(test_enc, test_labels_enc)

    # Load saved model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-reviews-genres"
    )

    trainer = Trainer(model=model)

    # ---------- Evaluation ----------
    print("Running evaluation...")
    trainer.evaluate(test_dataset)

    preds = trainer.predict(test_dataset)

    predicted_ids = preds.predictions.argmax(-1)
    predicted_labels = [id2label[i] for i in predicted_ids]

    print(classification_report(test_labels, predicted_labels))

    # ---------- Correct examples ----------
    print("\nCorrect Predictions:")
    for t, p, text in random.sample(list(zip(test_labels, predicted_labels, test_texts)), 20):
        if t == p:
            print("LABEL:", t)
            print("TEXT:", text[:100], "...\n")

    # ---------- Incorrect examples ----------
    print("\nIncorrect Predictions:")
    for t, p, text in random.sample(list(zip(test_labels, predicted_labels, test_texts)), 20):
        if t != p:
            print("TRUE:", t)
            print("PRED:", p)
            print("TEXT:", text[:100], "...\n")

    # ---------- Heatmap ----------
    counts = defaultdict(int)
    for t, p in zip(test_labels, predicted_labels):
        counts[(t, p)] += 1

    df = pd.DataFrame([
        {"True": t, "Pred": p, "Count": c}
        for (t, p), c in counts.items()
    ])

    table = df.pivot_table(index="True", columns="Pred", values="Count")

    plt.figure(figsize=(9, 7))
    sns.heatmap(table, cmap="Purples", linewidths=1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()