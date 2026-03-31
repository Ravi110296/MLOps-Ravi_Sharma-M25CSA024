from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    "facebook/deit-small-patch16-224",
    num_labels=100,
    ignore_mismatched_sizes=True
)

# Freeze backbone (like your training)
for param in model.vit.parameters():
    param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")