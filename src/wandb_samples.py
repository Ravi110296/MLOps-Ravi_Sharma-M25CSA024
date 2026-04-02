import wandb
import torch
import torch.nn as nn
from torchvision import models
from dataset_cifar10 import get_cifar10_loaders
import numpy as np

# ART
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, BasicIterativeMethod

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(project="dlops-assignment-q2")

# ================== MODEL ==================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("resnet18_cifar10.pth", map_location=device))
model = model.to(device)
model.eval()

# ================== DATA ==================
_, test_loader = get_cifar10_loaders(batch_size=10)
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# ================== FGSM (SCRATCH) ==================
images.requires_grad_(True)

outputs = model(images)
loss = nn.CrossEntropyLoss()(outputs, labels)

model.zero_grad()
loss.backward()

fgsm_imgs = images + 0.01 * images.grad.sign()
fgsm_imgs = torch.clamp(fgsm_imgs, -1, 1).detach()

# ================== ART CLASSIFIER ==================
classifier = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=None,
    input_shape=(3, 32, 32),
    nb_classes=10,
    clip_values=(-1.0, 1.0),
)

# Convert to numpy safely
x_np = images.detach().cpu().numpy()
y_np = labels.detach().cpu().numpy()

# ================== FGSM (ART) ==================
fgsm_art = FastGradientMethod(estimator=classifier, eps=0.01)
fgsm_art_imgs = torch.from_numpy(fgsm_art.generate(x=x_np)).to(device)

# ================== PGD (ART) ==================
pgd = ProjectedGradientDescent(
    estimator=classifier,
    eps=0.10,
    eps_step=0.01,
    max_iter=20
)
pgd_imgs = torch.from_numpy(pgd.generate(x=x_np, y=y_np)).to(device)

# ================== BIM (ART) ==================
bim = BasicIterativeMethod(
    estimator=classifier,
    eps=0.10,
    eps_step=0.01,
    max_iter=20
)
bim_imgs = torch.from_numpy(bim.generate(x=x_np, y=y_np)).to(device)

# ================== LOG ==================
log_data = {}
class_names = [str(l.item()) for l in labels]
for i in range(10):
    log_data[f"Sample_{i}"] = [
        wandb.Image(images[i].cpu(), caption=f"Clean (Label: {class_names[i]})"),
        wandb.Image(fgsm_imgs[i].cpu(), caption="FGSM"),
        wandb.Image(fgsm_art_imgs[i].cpu(), caption="FGSM_ART"),
        wandb.Image(pgd_imgs[i].cpu(), caption="PGD"),
        wandb.Image(bim_imgs[i].cpu(), caption="BIM"),
    ]

wandb.log(log_data)

wandb.finish()
print("✅ WandB logging complete!")