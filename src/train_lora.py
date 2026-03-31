import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model
from dataset import get_cifar100_loaders
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler

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

def train(rank,alpha):

    print(f"\n Running LoRA: rank={rank}, alpha={alpha}")

    train_loader, test_loader = get_cifar100_loaders()

    model = ViTForImageClassification.from_pretrained(
        "facebook/deit-small-patch16-224",
        num_labels=100,
        ignore_mismatched_sizes=True
    )
    scaler = GradScaler("cuda")

    # LoRA config
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,
        bias="none"
    )
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    for name, param in model.named_parameters():
        if "lora" not in name and "classifier" not in name:
            param.requires_grad = False

    model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4
    )

    criterion = nn.CrossEntropyLoss()   

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(3):
        model.train()
        total_loss = 0
        correct = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).logits
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = model(images).logits
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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

    df.to_csv(f"lora_results_r{rank}_a{alpha}.csv", index=False)

    # Loss plot
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.title(f"Loss (r={rank}, a={alpha})")
    plt.savefig(f"loss_r{rank}_a{alpha}.png")

    # Accuracy plot
    plt.figure()
    plt.plot(train_accs, label="train_acc")
    plt.plot(val_accs, label="val_acc")
    plt.legend()
    plt.title(f"Accuracy (r={rank}, a={alpha})")
    plt.savefig(f"acc_r{rank}_a{alpha}.png")

if __name__ == "__main__":
    # ranks = [2, 4, 8]
    # alphas = [2, 4, 8]

    # for rank in ranks:
    #     for alpha in alphas:
    #         print(f"Running experiment: r={rank}, alpha={alpha}")
    #         train(rank, alpha)
    train(2,2)