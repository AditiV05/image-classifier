"""
Slice 1 baseline: a single linear layer (logistic regression).
No hidden layers, no convolutions — this exists purely to get the
train/eval loop working and to establish a number worth beating.
"""

import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten: 28x28 image -> 784-length vector
        # Linear: 784 inputs -> 10 outputs (one score per class)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 10)
        )

    def forward(self, x):
        return self.net(x)  # raw logits, NOT probabilities
    
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Feature extraction: convolution + pooling ---
        self.features = nn.Sequential(
            # Conv layer 1: 1 input channel (grayscale) -> 16 filters, each 3x3
            # padding=1 keeps the output the same size as input (28x28)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),                    # nonlinearity — lets the network learn complex patterns
            nn.MaxPool2d(kernel_size=2),  # shrinks 28x28 -> 14x14, keeps strongest signal per 2x2 block

            # Conv layer 2: 16 inputs -> 32 filters
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # shrinks 14x14 -> 7x7
        )

        # --- Decision layer: same idea as the baseline, but fed real features ---
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # 32 channels x 7 x 7 -> flat vector
            nn.Linear(32 * 7 * 7, 10)      # 1568 inputs -> 10 class scores
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x