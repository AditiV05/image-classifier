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