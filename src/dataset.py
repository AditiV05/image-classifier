"""
Loads Fashion-MNIST, splits into train/val/test, and provides a helper
to visualize sample images.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def get_dataloaders(batch_size=64, val_size=10000, data_dir="../data", seed=42):
    """Downloads Fashion-MNIST (if needed) and returns train/val/test DataLoaders."""
    torch.manual_seed(seed)

    # ToTensor: PIL image -> tensor, scales pixels from [0,255] to [0.0,1.0]
    # Normalize: centers values around 0 (mean=0.5, std=0.5 -> roughly [-1, 1])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_val_data = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_data = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_len = len(train_val_data) - val_size
    train_data, val_data = random_split(train_val_data, [train_len, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_val_data


def show_samples(dataset, class_names=CLASS_NAMES, n=10):
    """Plots n random images with their labels — sanity check before training."""
    fig, axes = plt.subplots(1, n, figsize=(15, 2))
    indices = np.random.choice(len(dataset), n, replace=False)
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].set_title(class_names[label], fontsize=8)
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_loader, val_loader, test_loader, raw_train = get_dataloaders()
    print(f"Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")
    show_samples(raw_train)