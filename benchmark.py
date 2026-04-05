import time
import numpy as np
import torch
import torch.nn as nn

from models.MLPmodel import SoftmaxModel
from models.KAN import KAN
from models.FastKANmodel import FastKAN
from models.UA_KANmodel import UA_KAN
from models.DUQmodel import DUQ
from utils.functions import ActivationFunctions, model_stats
from datasets.load_datasets import load_D1

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (3,)

def get_train_data():
    _, train_loader, _, _ = load_D1(seed=1, path='./datasets')
    X, y = [], []
    for bx, by in train_loader:
        X.append(bx.numpy()); y.append(by.numpy())
    return np.concatenate(X), np.concatenate(y)


configs = {
    "MLP": lambda: SoftmaxModel([3, 32, 16, 3], activationFunction=nn.ReLU()),
    "KAN":      lambda: KAN([3,32,3], grid=4),
    "FastKAN": lambda: FastKAN([3, 32, 3], num_grids=4),
    "DUQ": lambda: DUQ(3, 32, 16, 3, 1e-2, 1.0),
    "UA-KAN": lambda: UA_KAN(
        [3, 32, 3], num_grids=4,
        base_activation=ActivationFunctions(gamma=4.0).RBF_SiLU,
        denominator=1.0
    ),
}

# print pytorch models first
for name, build in configs.items():
    print(f"\n{name}")
    model_stats(build().to(device), INPUT_SIZE, device)

X_train, y_train = get_train_data()

from tabpfn import TabPFNClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

print("\nTabNet")
clf = TabNetClassifier(verbose=0)
clf.fit(X_train, y_train)
model_stats(clf.network.to(device), INPUT_SIZE, device)

# ONLY PARAMS, its in-context learner every prediction runs a transformer over the entire training set
print("\nTabPFN")
clf = TabPFNClassifier()
clf.fit(X_train, y_train)
params = sum(p.numel() for p in clf.model_.parameters())
print(f"  Params: {params:,}")
