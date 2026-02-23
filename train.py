# train.py

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from data import get_data, MyDataset
from utils import get_tokenizer, build_label_maps, compute_metrics


def main():

    train_texts, train_labels, test_texts, test_labels = get_data()

    label2id, id2label = build_label_maps(train_labels)

    tokenizer = get_tokenizer()

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    test_enc = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

    train_labels_enc = [label2id[y] for y in train_labels]
    test_labels_enc = [label2id[y] for y in test_labels]

    train_dataset = MyDataset(train_enc, train_labels_enc)
    test_dataset = MyDataset(test_enc, test_labels_enc)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-cased",
        num_labels=len(id2label)
    )

    training_args = TrainingArguments(
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=10,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    learning_rate=5e-5,              # initial learning rate for Adam optimizer
    warmup_steps=100,                # number of warmup steps for learning rate scheduler (set lower because of small dataset size)
    weight_decay=0.01,               # strength of weight decay
    output_dir='./results',          # output directory
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,               # number of steps to output logging (set lower because of small dataset size)
    eval_strategy='steps',     # evaluate during fine-tuning so that we can see progress
    report_to=[],  # Disables wandb logging
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model("distilbert-reviews-genres")


if __name__ == "__main__":
    main()