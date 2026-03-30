import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from dataset import get_cifar100_loaders
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def train():

    train_loader, test_loader = get_cifar100_loaders()

    model = ViTForImageClassification.from_pretrained(
        "google/vit-small-patch16-224",
        num_labels=100
    )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

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

        acc = correct / len(train_loader.dataset)

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Acc: {acc:.4f}")

if __name__ == "__main__":
    train()