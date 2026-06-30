"""
Slice 1: train the baseline model and report an honest test accuracy
plus a per-class breakdown.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

from dataset import get_dataloaders, CLASS_NAMES
from model import BaselineModel

BATCH_SIZE = 64
EPOCHS = 5
LR = 0.01


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct = 0, 0

    for images, labels in loader:
        optimizer.zero_grad()              # clear gradients from previous batch
        outputs = model(images)            # forward pass
        loss = criterion(outputs, labels)  # how wrong are we
        loss.backward()                    # compute gradients
        optimizer.step()                   # update weights

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():  # no backprop needed during eval
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main():
    train_loader, val_loader, test_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

    model = BaselineModel()
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch}/{EPOCHS}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest accuracy: {test_acc:.3%}")

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images).argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("\nPer-class report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))


if __name__ == "__main__":
    main()