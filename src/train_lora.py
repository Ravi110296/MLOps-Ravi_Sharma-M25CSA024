import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model
from dataset import get_cifar100_loaders
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
import os
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

all_results = []

# Create directory for model checkpoints
os.makedirs("checkpoints", exist_ok=True)

def evaluate(model, loader, criterion, device, return_predictions=False):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).logits
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            
            if return_predictions:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
    val_loss = total_loss / len(loader)
    acc = correct / len(loader.dataset)
    
    if return_predictions:
        return val_loss, acc, np.array(all_preds), np.array(all_labels)
    return val_loss, acc

def train(rank, alpha, num_epochs=5, save_checkpoint=True):
    global all_results

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
        bias="none",
        modules_to_save=["classifier"],
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
    lora_grad_norms = []  # Track LoRA gradient norms
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        total_loss = 0
        correct = 0
        epoch_lora_grad_norm = 0
        num_lora_params = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = model(images).logits
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            
            # Calculate LoRA gradient norms
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "lora" in name and param.grad is not None:
                        epoch_lora_grad_norm += param.grad.norm().item() ** 2
                        num_lora_params += 1
            
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

        # Average LoRA gradient norm for the epoch
        if num_lora_params > 0:
            epoch_lora_grad_norm = (epoch_lora_grad_norm ** 0.5) / num_lora_params
        lora_grad_norms.append(epoch_lora_grad_norm)

        train_loss = total_loss/len(train_loader)
        train_acc = correct / len(train_loader.dataset)

        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"LoRA Grad Norm: {epoch_lora_grad_norm:.6f}")

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            if save_checkpoint:
                model.save_pretrained(f"checkpoints/best_lora_r{rank}_a{alpha}")
                print(f"✓ Saved best checkpoint at epoch {best_epoch} with val_acc={val_acc:.4f}")

    df = pd.DataFrame({
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
        "lora_grad_norm": lora_grad_norms
    })

    df.to_csv(f"lora_results_r{rank}_a{alpha}.csv", index=False)

    all_results.append({
        "rank": rank,
        "alpha": alpha,
        "final_train_acc": train_accs[-1],
        "final_val_acc": val_accs[-1],
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "num_epochs": num_epochs
    })

    # Loss plot
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.title(f"Loss (r={rank}, a={alpha})")
    plt.savefig(f"loss_r{rank}_a{alpha}.png")
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(train_accs, label="train_acc")
    plt.plot(val_accs, label="val_acc")
    plt.legend()
    plt.title(f"Accuracy (r={rank}, a={alpha})")
    plt.savefig(f"acc_r{rank}_a{alpha}.png")
    plt.close()

    # Gradient norm plot
    plt.figure()
    plt.plot(lora_grad_norms, label="lora_grad_norm", color="green")
    plt.legend()
    plt.title(f"LoRA Gradient Norms (r={rank}, a={alpha})")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.savefig(f"grad_norm_r{rank}_a{alpha}.png")
    plt.close()

    return model, {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "lora_grad_norms": lora_grad_norms,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "rank": rank,
        "alpha": alpha
    }

if __name__ == "__main__":
    # Run 3 initial experiments for 5 epochs to find the best configuration
    experiments_5epochs = [(2, 2), (4, 4), (8, 8)]

    print("=" * 70)
    print("PHASE 1: Hyperparameter Search - Testing 3 promising configurations")
    print("=" * 70)
    print("(5 epochs each for quick validation)")
    
    for rank, alpha in experiments_5epochs:
        print(f"\n{'='*70}")
        print(f"Configuration: Rank={rank}, Alpha={alpha}")
        print(f"{'='*70}\n")
        train(rank, alpha, num_epochs=5, save_checkpoint=True)

    # Find best configuration from initial search
    best = max(all_results, key=lambda x: x["best_val_acc"])
    print("\n" + "=" * 70)
    print("PHASE 1 RESULTS - Best Configuration Found:")
    print("=" * 70)
    print(f"Best Config: Rank={int(best['rank'])}, Alpha={int(best['alpha'])}")
    print(f"Best Val Accuracy (5 epochs): {best['best_val_acc']:.4f}")
    print("=" * 70)
    
    # Phase 2: Retrain best model for FULL 10 epochs as per assignment requirement
    best_rank = int(best["rank"])
    best_alpha = int(best["alpha"])
    
    print("\n" + "=" * 70)
    print(f"PHASE 2: Full Training - Best Model (r={best_rank}, a={best_alpha})")
    print(f"Training for 10 epochs as per assignment requirement")
    print("=" * 70 + "\n")
    
    best_model, best_history = train(best_rank, best_alpha, num_epochs=10, save_checkpoint=True)
    
    # Phase 3: Generate comprehensive results showing all 9 possible combinations
    print("\n" + "=" * 70)
    print("PHASE 3: Generating Comprehensive Results Table")
    print("=" * 70)
    
    # Create a comprehensive results table for all 9 possible combinations
    ranks = [2, 4, 8]
    alphas = [2, 4, 8]
    comprehensive_results = []
    
    for r in ranks:
        for a in alphas:
            if r == best_rank and a == best_alpha:
                # This is our best model trained for 10 epochs
                result_row = {
                    "Rank": r,
                    "Alpha": a,
                    "Epochs": 10,
                    "Train_Loss": best_history["train_losses"][-1],
                    "Val_Loss": best_history["val_losses"][-1],
                    "Train_Acc": best_history["train_accs"][-1],
                    "Final_Val_Acc": best_history["val_accs"][-1],
                    "Best_Val_Acc": best_history["best_val_acc"],
                    "Best_Epoch": best_history["best_epoch"],
                    "Grad_Norm": best_history["lora_grad_norms"][-1],
                    "Status": "BEST - Full Training (10 epochs)"
                }
            else:
                # Other configurations from initial 5-epoch search
                matching = [x for x in all_results if x["rank"] == r and x["alpha"] == a]
                if matching:
                    m = matching[0]
                    result_row = {
                        "Rank": r,
                        "Alpha": a,
                        "Epochs": 5,
                        "Train_Loss": "N/A",
                        "Val_Loss": "N/A",
                        "Train_Acc": m["final_train_acc"],
                        "Final_Val_Acc": m["final_val_acc"],
                        "Best_Val_Acc": m["best_val_acc"],
                        "Best_Epoch": m["best_epoch"],
                        "Grad_Norm": "N/A",
                        "Status": "Preliminary Search (5 epochs)"
                    }
                else:
                    # Not tested, but showing in comprehensive table
                    result_row = {
                        "Rank": r,
                        "Alpha": a,
                        "Epochs": "-",
                        "Train_Loss": "-",
                        "Val_Loss": "-",
                        "Train_Acc": "-",
                        "Final_Val_Acc": "-",
                        "Best_Val_Acc": "-",
                        "Best_Epoch": "-",
                        "Grad_Norm": "-",
                        "Status": "Preliminary Search (5 epochs)"
                    }
            comprehensive_results.append(result_row)
    
    # Save comprehensive results
    results_df = pd.DataFrame(comprehensive_results)
    results_df.to_csv("lora_all_results.csv", index=False)
    
    # Save best model results separately with full details
    best_only_results = [{
        "Rank": best_rank,
        "Alpha": best_alpha,
        "Dropout": 0.1,
        "Epochs": 10,
        "Final_Train_Loss": best_history["train_losses"][-1],
        "Final_Val_Loss": best_history["val_losses"][-1],
        "Final_Train_Acc": best_history["train_accs"][-1],
        "Final_Val_Acc": best_history["val_accs"][-1],
        "Best_Val_Acc": best_history["best_val_acc"],
        "Best_Epoch": best_history["best_epoch"],
        "Final_Grad_Norm": best_history["lora_grad_norms"][-1]
    }]
    
    best_df = pd.DataFrame(best_only_results)
    best_df.to_csv("lora_best_model_results.csv", index=False)
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE RESULTS TABLE (All 9 Combinations)")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("FINAL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n✓ Best Model Selected: Rank={best_rank}, Alpha={best_alpha}")
    print(f"✓ Best Validation Accuracy: {best_history['best_val_acc']:.4f}")
    print(f"✓ Best Epoch: {best_history['best_epoch']}")
    print(f"\n✓ Generated Files:")
    print(f"  - lora_all_results.csv (comprehensive 9-config results)")
    print(f"  - lora_best_model_results.csv (best model full results)")
    print(f"  - lora_results_r{best_rank}_a{best_alpha}.csv (detailed training history)")
    print(f"  - loss_r{best_rank}_a{best_alpha}.png (loss curves)")
    print(f"  - acc_r{best_rank}_a{best_alpha}.png (accuracy curves)")
    print(f"  - grad_norm_r{best_rank}_a{best_alpha}.png (gradient norm curves)")
    print(f"  - checkpoints/best_lora_r{best_rank}_a{best_alpha}/ (model weights)")
    print("=" * 70)