"""
Evaluate best model and generate visualizations for Step 3
- Class-wise test accuracy histogram
- Comparison of all experiments
- Test results table
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from peft import PeftConfig, PeftModel
from dataset import get_cifar100_loaders
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

try:
    from peft import AutoPeftModelForImageClassification
except ImportError:
    AutoPeftModelForImageClassification = None

device = "cuda" if torch.cuda.is_available() else "cpu"

# CIFAR-100 class names (100 classes)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'champ', 'change', 'chap',
    'char', 'chase', 'chat', 'cheap', 'cheat', 'check', 'cheese', 'cheetah',
    'chef', 'cherry', 'chest', 'chicken', 'chickpea', 'chihuahua', 'child', 'chime',
    'chimpanzee', 'chin', 'china', 'chinaware', 'chinese', 'chink', 'chino', 'chip',
    'chipmunk', 'chirp', 'chit', 'chive', 'chock', 'choice', 'choir', 'choke',
    'choker', 'chomp', 'chooser', 'chop', 'chopper', 'choppy', 'chops', 'chopstick',
    'choral', 'chord', 'chore', 'chorea', 'choreo', 'chorial', 'choric', 'chorion',
    'chorister', 'chorizo', 'choroid', 'chortle', 'chorus', 'chose', 'chosen', 'chough',
    'chouse', 'choux', 'chow', 'chowder', 'chub', 'chuck', 'chuckle', 'chug',
    'chukka', 'chump', 'chunk', 'chunky', 'church', 'churl', 'churly', 'churn',
    'churro', 'chute', 'chutney', 'chutzpah'
]


def get_class_wise_accuracy(y_true, y_pred, num_classes=100):
    """Calculate per-class accuracy"""
    class_acc = {}
    for class_idx in range(num_classes):
        class_mask = y_true == class_idx
        if class_mask.sum() > 0:
            class_acc[class_idx] = ((y_pred[class_mask] == y_true[class_mask]).sum() / class_mask.sum().item()).item()
        else:
            class_acc[class_idx] = 0.0
    return class_acc


def evaluate_with_predictions(model_path, rank, alpha):
    """Evaluate model and get predictions"""
    print(f"\nEvaluating model: r={rank}, a={alpha}")
    print("-" * 50)
    
    # Load best model (supports both newer and older PEFT versions)
    if AutoPeftModelForImageClassification is not None:
        model = AutoPeftModelForImageClassification.from_pretrained(model_path, is_trainable=True)
    else:
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = ViTForImageClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=100,
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=True)
    model.to(device)
    model.eval()
    
    # Get data loaders
    train_loader, test_loader = get_cifar100_loaders()
    
    # Get test predictions
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).logits
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    overall_acc = (all_preds == all_labels).mean()
    class_acc = get_class_wise_accuracy(torch.tensor(all_labels), torch.tensor(all_preds))
    
    # Count trainable parameters used by LoRA + saved trainable modules
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Overall Test Accuracy: {overall_acc:.4f}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Total Parameters: {total_params:,}")
    
    return all_preds, all_labels, class_acc, overall_acc, trainable_params, total_params


def plot_class_wise_accuracy_histogram(class_acc, rank, alpha, top_k=20):
    """Plot histogram of class-wise accuracy"""
    class_accs = list(class_acc.values())
    class_indices = list(class_acc.keys())
    
    # Sort by accuracy
    sorted_indices = sorted(range(len(class_accs)), key=lambda i: class_accs[i])
    
    # Get worst 10 and best 10 classes
    worst_10_idx = sorted_indices[:10]
    best_10_idx = sorted_indices[-10:]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Worst 10 classes
    worst_classes = [CIFAR100_CLASSES[i] if i < len(CIFAR100_CLASSES) else f"Class {i}" for i in worst_10_idx]
    worst_accs = [class_accs[i] for i in worst_10_idx]
    axes[0].bar(worst_classes, worst_accs, color='red', alpha=0.7)
    axes[0].set_title(f'Worst 10 Classes (r={rank}, a={alpha})', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim([0, 1])
    axes[0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(worst_accs):
        axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Best 10 classes
    best_classes = [CIFAR100_CLASSES[i] if i < len(CIFAR100_CLASSES) else f"Class {i}" for i in best_10_idx]
    best_accs = [class_accs[i] for i in best_10_idx]
    axes[1].bar(best_classes, best_accs, color='green', alpha=0.7)
    axes[1].set_title(f'Best 10 Classes (r={rank}, a={alpha})', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim([0, 1])
    axes[1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(best_accs):
        axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"class_wise_accuracy_r{rank}_a{alpha}.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: class_wise_accuracy_r{rank}_a{alpha}.png")
    plt.close()
    
    # Full histogram
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(class_accs)), sorted(class_accs), color='skyblue', alpha=0.7)
    ax.set_title(f'Per-Class Test Accuracy Distribution (r={rank}, a={alpha})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class (sorted by accuracy)')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1])
    ax.axhline(y=np.mean(class_accs), color='r', linestyle='--', label=f'Mean: {np.mean(class_accs):.4f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"class_wise_accuracy_all_r{rank}_a{alpha}.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: class_wise_accuracy_all_r{rank}_a{alpha}.png")
    plt.close()


def create_test_results_table(results_list, output_file="test_results_table.csv"):
    """Create table with test results for all configurations"""
    df = pd.DataFrame(results_list)
    df = df[['lora_with_without', 'rank', 'alpha', 'dropout', 
             'overall_test_accuracy', 'trainable_parameters', 'total_parameters']]
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    print(df.to_string(index=False))
    return df


def main():
    """Main evaluation routine"""
    print("=" * 60)
    print("STEP 3: Evaluation & Visualization")
    print("=" * 60)
    
    # Configuration
    best_config = (8, 8)  # best model
    best_model_path = f"checkpoints/best_lora_r{best_config[0]}_a{best_config[1]}"
    
    # Evaluate best model
    try:
        preds, labels, class_acc, overall_acc, trainable_params, total_params = evaluate_with_predictions(
            best_model_path, 
            best_config[0], 
            best_config[1]
        )
        
        # Generate visualizations
        plot_class_wise_accuracy_histogram(class_acc, best_config[0], best_config[1])
        
        # Create test results table
        test_results = [{
            'lora_with_without': 'With LoRA',
            'rank': best_config[0],
            'alpha': best_config[1],
            'dropout': 0.1,
            'overall_test_accuracy': overall_acc,
            'trainable_parameters': trainable_params,
            'total_parameters': total_params
        }]
        
        create_test_results_table(test_results)
        
        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: Model not found at {best_model_path}")
        print(f"Make sure training is complete and model is saved.")
        print(e)


if __name__ == "__main__":
    main()
