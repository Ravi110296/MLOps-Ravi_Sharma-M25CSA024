import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_cifar100_loaders(batch_size=128):

    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    train_dataset = Subset(train_dataset, range(10000))

    test_dataset = datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    test_dataset = Subset(test_dataset, range(2000))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader