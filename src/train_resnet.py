import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset_cifar10 import get_cifar10_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, test_loader = get_cifar10_loaders(batch_size=128)

# Model (from scratch)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 → 10 classes
model = model.to(device)

# Loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Scheduler (important for performance)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 20

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {running_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.4f} "
          f"Test Acc: {test_acc:.4f}")

# Save model
torch.save(model.state_dict(), "resnet18_cifar10.pth")

print("Training complete and model saved!")