# Fashion-MNIST Image Classifier

End-to-end image classifier trained from scratch in PyTorch — no pre-trained models, no external APIs.
Train → evaluate honestly → serve via a self-built API → demo in browser.

## Status

🚧 In progress — Slice 2 (CNN model)

## Setup

## Project structure

- `src/dataset.py` — loads Fashion-MNIST, splits into train/val/test, visualizes samples
- `src/model.py` — model architectures (baseline linear model, CNN)
- `src/train.py` — training and evaluation loop
- `models/` — saved trained weights (not committed)
- `data/` — Fashion-MNIST, auto-downloaded (not committed)

## Results

### Baseline (single linear layer, 7,850 parameters)

A single `Linear(784, 10)` layer with no hidden layers or convolutions — exists to validate the train/eval loop and set a number to beat.

- **Test accuracy: 82.84%**
- Strong classes: Trouser (0.96 f1), Bag (0.93), Ankle boot (0.92)
- Weak classes: Shirt (0.56 f1), Pullover (0.73), Coat (0.73)

**Why the weak classes struggle:** a linear layer treats every pixel independently — it has no concept of neighboring pixels forming a shape. Shirt, Pullover, Coat, and T-shirt/top all share a similar upper-body silhouette in a 28×28 grayscale thumbnail, and differ mainly in subtle local details (collar shape, sleeve length) that a linear model structurally cannot capture.

### CNN (two conv layers + max pooling, 20,490 parameters)

Adds convolution layers before the final decision layer, so the model learns local pattern detectors (edges, curves) instead of treating pixels independently.

- **Test accuracy: TBD**

## What's next

- Slice 3: proper confusion matrix and failure analysis
- Slice 4: data augmentation / regularization to fight overfitting
- Slice 5: serve via FastAPI
- Slice 6: web demo
