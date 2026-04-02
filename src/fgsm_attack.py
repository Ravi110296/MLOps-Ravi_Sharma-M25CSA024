import torch
import torch.nn as nn
from torchvision import models
from dataset_cifar10 import get_cifar10_loaders
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("resnet18_cifar10.pth"))
model = model.to(device)
model.eval()

# Data
_, test_loader = get_cifar10_loaders(batch_size=128)

criterion = nn.CrossEntropyLoss()

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    return perturbed_image


def test_fgsm(epsilon):
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        data_grad = images.grad.data
        perturbed_images = fgsm_attack(images, epsilon, data_grad)

        outputs = model(perturbed_images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


epsilons = [0, 0.01, 0.03, 0.05, 0.1]

results = []

print("FGSM Attack Results:")
for eps in epsilons:
    acc = test_fgsm(eps)
    print(f"Epsilon: {eps:.2f} → Accuracy: {acc:.4f}")
    results.append({"epsilon": eps,"accuracy": acc})

# Save CSV
df = pd.DataFrame(results)
df.to_csv("fgsm_results.csv", index=False)

print("✅ Results saved to fgsm_results.csv")
