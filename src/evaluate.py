"""
Slice 3: confusion matrix and failure analysis.
Loads saved model weights — no retraining needed.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import get_dataloaders, CLASS_NAMES
from model import CNNModel


def get_all_preds_and_labels(model, loader):
    all_preds, all_labels, all_images = [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images).argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_images.extend(images)
    return all_preds, all_labels, all_images


def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion Matrix — CNN on Fashion-MNIST test set")
    plt.tight_layout()
    plt.savefig("../confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")
    plt.show()


def plot_failures(images, labels, preds, class_names, n=10):
    """Show n images the model got wrong, with true vs predicted label."""
    wrong_indices = [i for i, (l, p) in enumerate(zip(labels, preds)) if l != p]
    sample = np.random.choice(wrong_indices, min(n, len(wrong_indices)), replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.flatten()
    for i, idx in enumerate(sample):
        img = images[idx].squeeze().numpy()
        true_label = class_names[labels[idx]]
        pred_label = class_names[preds[idx]]
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=8, color="red")
        axes[i].axis("off")
    plt.suptitle("Failure cases — images the model got wrong", fontsize=12)
    plt.tight_layout()
    plt.savefig("../failure_cases.png", dpi=150)
    print("Saved failure_cases.png")
    plt.show()


def main():
    _, _, test_loader, _ = get_dataloaders()

    model = CNNModel()
    model.load_state_dict(torch.load("../models/cnn_model.pt", weights_only=True))
    print("Loaded model weights from models/cnn_model.pt")

    preds, labels, images = get_all_preds_and_labels(model, test_loader)

    plot_confusion_matrix(labels, preds, CLASS_NAMES)
    plot_failures(images, labels, preds, CLASS_NAMES)


if __name__ == "__main__":
    main()