import torch
import torch.nn as nn
import torch.nn.functional as F


class MCDropoutMLP(nn.Module):
    def __init__(self, architecture, dropout_p=0.2):
        super().__init__()
        layers = []
        for i in range(len(architecture) - 1):
            layers.append(nn.Linear(architecture[i], architecture[i + 1]))
            if i < len(architecture) - 2:  # no activation/dropout on output layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout_p))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def mc_forward(self, x, n_samples=20):
        """Keep dropout active at inference for n_samples stochastic passes."""
        self.train()  # enables dropout
        with torch.no_grad():
            samples = torch.stack([F.softmax(self.model(x), dim=1) for _ in range(n_samples)])
        self.eval()
        return samples  # [n_samples, batch, n_classes]
