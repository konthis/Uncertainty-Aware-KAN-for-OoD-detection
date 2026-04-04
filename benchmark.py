import torch
from models.MLPmodel import SoftmaxModel
from models.KAN import KAN
from models.FastKANmodel import FastKAN
from models.UA_KANmodel import UA_KAN 
from models.DUQmodel import DUQ
from utils.functions import ActivationFunctions, model_stats
import time
import numpy as np
from datasets.load_datasets import load_D1



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (3,)

configs = {
    # specific import
    "MLP":      lambda: SoftmaxModel([3, 32, 16, 3], activationFunction=__import__('torch.nn', fromlist=['ReLU']).ReLU()),
    "KAN":      lambda: KAN([3,32,3], grid=4),
    "FastKAN":  lambda: FastKAN([3, 32, 3], num_grids=4),
    "DUQ":      lambda: DUQ(3, 32, 16, 3, 1e-2, 1.0),
    "UA-KAN":   lambda: UA_KAN(
                    [3, 32, 3], num_grids=4,
                    base_activation=ActivationFunctions(gamma=4.0).RBF_SiLU,
                    denominator=1.0
                ),
}

# for name, build in configs.items():
    # print(f"\n{name}")
    # model = build().to(device)
    # model_stats(model, INPUT_SIZE, device)


    # ── Inference time + param count for sklearn-style models ───────────────────
def sklearn_model_stats(clf, X_sample: np.ndarray, n_warmup: int = 50, n_runs: int = 500):
    # Warm up
    for _ in range(n_warmup):
        clf.predict_proba(X_sample)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        clf.predict_proba(X_sample)
    ms = (time.perf_counter() - t0) / n_runs * 1000

    # Param count — TabNet exposes its network, TabPFN is a frozen foundation model
    try:
        params = sum(p.numel() for p in clf.network.parameters() if p.requires_grad)
    except AttributeError:
        params = None

    print(f"  Params (trainable): {params if params is not None else 'N/A (foundation model)'}")
    print(f"  Inference time:     {ms:.4f} ms/sample")
    return {'params': params, 'ms': ms}


# ── Load a small training split for fitting sklearn models ──────────────────
def get_train_data():
    _, train_loader, _, _ = load_D1(seed=1, path='./datasets')
    X, y = [], []
    for bx, by in train_loader:
        X.append(bx.numpy()); y.append(by.numpy())
    return np.concatenate(X), np.concatenate(y)


print("\nLoading training data for sklearn models...")
X_train, y_train = get_train_data()
X_single = X_train[:1]   # single sample for timing

sklearn_configs = {
    "TabPFN": lambda: __import__('tabpfn', fromlist=['TabPFNClassifier']).TabPFNClassifier(),
    "TabNet": lambda: __import__('pytorch_tabnet.tab_model', fromlist=['TabNetClassifier']).TabNetClassifier(verbose=0),
}

for name, build in sklearn_configs.items():
    print(f"\n{name}")
    clf = build()
    clf.fit(X_train, y_train)
    sklearn_model_stats(clf, X_single)
