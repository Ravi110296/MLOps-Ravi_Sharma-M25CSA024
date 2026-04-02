import torch
import torch.nn as nn
from torchvision import models
from dataset_cifar10 import get_cifar10_loaders
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
import numpy as np
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

# Loss + optimizer (needed for ART wrapper)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ART classifier
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=10,
    clip_values=(-1, 1),
)

# Convert test data to numpy
x_test = []
y_test = []

for images, labels in test_loader:
    x_test.append(images.numpy())
    y_test.append(labels.numpy())

x_test = np.concatenate(x_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

# FGSM attack
epsilons = [0.0, 0.01, 0.03, 0.05, 0.1]

results = []

print("FGSM (ART) Results:")
for eps in epsilons:
    attack = FastGradientMethod(estimator=classifier, eps=eps)
    x_adv = attack.generate(x=x_test)

    preds = classifier.predict(x_adv)
    acc = np.mean(np.argmax(preds, axis=1) == y_test)

    print(f"Epsilon: {eps:.2f} → Accuracy: {acc:.4f}")

    results.append({
        "epsilon": eps,
        "accuracy": acc
    })

# Save CSV
df = pd.DataFrame(results)
df.to_csv("fgsm_art_results.csv", index=False)

print("✅ ART FGSM results saved!")