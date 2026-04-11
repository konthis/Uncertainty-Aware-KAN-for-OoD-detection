import time
import numpy as np
import torch
import torch.nn as nn

from models.MLPmodel import SoftmaxModel, MLP
from models.KAN import KAN
from models.FastKANmodel import FastKAN
from models.UA_KANmodel import UA_KAN
from models.DUQmodel import DUQ
from models.MCDropoutModel import MCDropoutMLP
from utils.functions import ActivationFunctions, model_stats
from datasets.load_datasets import loadAllDataloaders
from xgboost import XGBClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARCH   = [3, 32, 16, 3]
INPUT_SIZE = (3,)


def get_train_data(dataset='ambrosia'):
    train_loader, _, *_ = loadAllDataloaders('./datasets', dataset=dataset)
    X, y = [], []
    for bx, by in train_loader:
        X.append(bx.numpy()); y.append(by.numpy())
    return np.concatenate(X), np.concatenate(y)


configs = {
    "MLP":         lambda: SoftmaxModel(ARCH, activationFunction=nn.ReLU()),
    "EnergyMLP":   lambda: MLP(ARCH, activationFunction=nn.ReLU()),
    "MCDropout":   lambda: MCDropoutMLP(ARCH, dropout_p=0.2),
    "KAN":         lambda: KAN([3, 32, 3], grid=4),
    "FastKAN":     lambda: FastKAN([3, 32, 3], num_grids=4),
    "DUQ":         lambda: DUQ(3, 32, 16, 3, 1e-2, 1.0),
    "UA-KAN":      lambda: UA_KAN(
        [3, 32, 3], num_grids=4,
        base_activation=ActivationFunctions(gamma=4.0).RBF_SiLU,
        denominator=1.0
    ),
}

# PyTorch models
for name, build in configs.items():
    print(f"\n{name}")
    model_stats(build().to(device), INPUT_SIZE, device)

# FT-Transformer
try:
    from train_FTTransformer import FTTransformerWrapper
    print("\nFT-Transformer")
    model_stats(FTTransformerWrapper(3, 3).to(device), INPUT_SIZE, device)
except Exception as e:
    print(f"\nFT-Transformer: skipped ({e})")

X_train, y_train = get_train_data()

print("\nTabNet")
from pytorch_tabnet.tab_model import TabNetClassifier

clf = TabNetClassifier(verbose=0)
clf.fit(X_train, y_train)
clf.network.to(device).eval()
model_stats(clf.network, (3,), device)


n_warmup, n_runs = 5, 20

# XGBoost — no FLOPs (tree-based), report n_estimators + inference time
print("\nXGBoost")
clf_xgb = XGBClassifier(n_estimators=300, max_depth=3, verbosity=0)
clf_xgb.fit(X_train, y_train)
X_test_np = X_train[:1]
for _ in range(n_warmup):
    clf_xgb.predict_proba(X_test_np)
t0 = time.perf_counter()
for _ in range(n_runs):
    clf_xgb.predict_proba(X_test_np)
ms = (time.perf_counter() - t0) / n_runs * 1000

from utils.functions import xgboost_flops  # or inline it

clf_xgb.fit(X_train, y_train)
flops = xgboost_flops(clf_xgb)
print(f"  Approx FLOPs/sample: {flops:,}")

print(f"  Trees: {clf_xgb.n_estimators}")
print(f"  Inference time: {ms:.4f} ms/sample")





# # TabPFN
# print("\nTabPFN")
# try:
#     from tabpfn import TabPFNClassifier
#     clf_pfn = TabPFNClassifier()
#     clf_pfn.fit(X_train, y_train)
#     params = sum(p.numel() for p in clf_pfn.model_.parameters())
#     print(f"  Params: {params:,}")
#     X_test_dummy = X_train[:1]
#     for _ in range(n_warmup):
#         clf_pfn.predict(X_test_dummy)
#     t0 = time.perf_counter()
#     for _ in range(n_runs):
#         clf_pfn.predict(X_test_dummy)
#     ms = (time.perf_counter() - t0) / n_runs * 1000
#     print(f"  Inference time: {ms:.4f} ms/sample")
#     print(f"  Note: in-context learner, FLOPs scale with training set size")
# except Exception as e:
#     print(f"  skipped ({e})")
