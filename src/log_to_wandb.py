import wandb
import pandas as pd

wandb.init(project="dlops-assignment-5", name="lora_r8_a8_best")

df = pd.read_csv("lora_results_best_r8_a8.csv")

for i in range(len(df)):
    wandb.log({
        "epoch": df["epoch"][i],
        "train_loss": df["train_loss"][i],
        "val_loss": df["val_loss"][i],
        "train_acc": df["train_acc"][i],
        "val_acc": df["val_acc"][i],
        "lora_grad_norm": df["lora_grad_norm"][i],
    })

wandb.finish()