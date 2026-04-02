# ===================================================================
# WandB 10 Samples - Clean + FGSM(scratch) + FGSM(ART) + PGD + BIM
# ===================================================================

import torch
import torch.nn as nn
from torchvision import models
from dataset_cifar10 import get_cifar10_loaders
import numpy as np
import wandb
from PIL import Image
import torchvision.transforms.functional as F
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, BasicIterativeMethod

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(
    project="dlops-ass5-q2-adversarial-detection",
    name="10_samples_FGSM_PGD_BIM",
    config={"eps": 0.03}
)

# ================== LOAD CLASSIFIER ==================
base_model = models.resnet18(pretrained=False)
base_model.fc = nn.Linear(base_model.fc.in_features, 10)
base_model.load_state_dict(torch.load("resnet18_cifar10.pth", map_location=device))
base_model = base_model.to(device)
base_model.eval()

classifier = PyTorchClassifier(
    model=base_model,
    loss=nn.CrossEntropyLoss(),
    optimizer=None,
    input_shape=(3, 32, 32),
    nb_classes=10,
    clip_values=(-1.0, 1.0),
)

# ================== ATTACKS ==================


fgsm_art = FastGradientMethod(estimator=classifier, eps=0.03)
pgd_art  = ProjectedGradientDescent(estimator=classifier, eps=0.03, eps_step=0.01, max_iter=10)
bim_art  = BasicIterativeMethod(estimator=classifier, eps=0.03, eps_step=0.01, max_iter=10)

def fgsm_scratch(images, labels, epsilon=0.03):
    images = images.clone().detach().to(device)
    images.requires_grad_(True)
    outputs = base_model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels.to(device))
    base_model.zero_grad()
    loss.backward()
    adv = torch.clamp(images + epsilon * images.grad.sign(), -1, 1)
    return adv.detach()

# ================== GET 10 IMAGES ==================
_, test_loader = get_cifar10_loaders(batch_size=128)
images, labels = next(iter(test_loader))
images = images[:10].to(device)
labels = labels[:10].to(device)

# Generate all attacks
adv_fgsm_scratch = fgsm_scratch(images, labels)
adv_fgsm_art     = torch.from_numpy(fgsm_art.generate(images.cpu().numpy(), labels.cpu().numpy())).float().to(device)
adv_pgd          = torch.from_numpy(pgd_art.generate(images.cpu().numpy(), labels.cpu().numpy())).float().to(device)
adv_bim          = torch.from_numpy(bim_art.generate(images.cpu().numpy(), labels.cpu().numpy())).float().to(device)

# ================== CREATE BIG GRID IMAGE ==================
def tensor_to_pil(t):
    t = torch.clamp((t + 1) / 2, 0, 1)          # [-1,1] → [0,1]
    t = (t * 255).byte().cpu()
    return F.to_pil_image(t)

grid_rows = []
for i in range(10):
    row = [
        tensor_to_pil(images[i]),
        tensor_to_pil(adv_fgsm_scratch[i]),
        tensor_to_pil(adv_fgsm_art[i]),
        tensor_to_pil(adv_pgd[i]),
        tensor_to_pil(adv_bim[i])
    ]
    grid_rows.append(row)

# Concatenate into one big image (10 rows x 5 columns)
grid_images = [Image.fromarray(np.hstack([np.array(img) for img in row])) for row in grid_rows]
final_grid = Image.fromarray(np.vstack([np.array(img) for img in grid_images]))

# Log to WandB
# Optional: Also log as table (may or may not render nicely)
table = wandb.Table(columns=["Clean", "FGSM_Scratch", "FGSM_ART", "PGD_ART", "BIM_ART"])
for i in range(10):
    table.add_data(
        wandb.Image((images[i] + 1)/2),
        wandb.Image((adv_fgsm_scratch[i] + 1)/2),
        wandb.Image((adv_fgsm_art[i] + 1)/2),
        wandb.Image((adv_pgd[i] + 1)/2),
        wandb.Image((adv_bim[i] + 1)/2)
    )
wandb.log({"10_Samples_Table": table})
wandb.finish()