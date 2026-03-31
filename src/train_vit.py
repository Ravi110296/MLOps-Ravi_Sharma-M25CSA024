import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from dataset import get_cifar100_loaders
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).logits
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    acc = correct / len(loader.dataset)
    return total_loss, acc

def train():


    train_loader, test_loader = get_cifar100_loaders()

    model = ViTForImageClassification.from_pretrained(
        "facebook/deit-small-patch16-224",
        num_labels=100,
        ignore_mismatched_sizes=True
    )
    for param in model.vit.parameters():
        param.requires_grad = False

    model.to(device)

    optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4
)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(10):
        model.train()
        total_loss = 0
        correct = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).logits
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

        train_loss = total_loss
        train_acc = correct / len(train_loader.dataset)

        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    df = pd.DataFrame({
    "epoch": list(range(1, 11)),
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_acc": train_accs,
    "val_acc": val_accs
})

    df.to_csv("baseline_results.csv", index=False)

    # Loss plot
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.savefig("loss.png")

    # Accuracy plot
    plt.figure()
    plt.plot(train_accs, label="train_acc")
    plt.plot(val_accs, label="val_acc")
    plt.legend()
    plt.savefig("accuracy.png")

if __name__ == "__main__":
    train()