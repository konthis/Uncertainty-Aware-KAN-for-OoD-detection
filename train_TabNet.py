import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.load_datasets import load_D1, createSklearnDataloader, GaussianNoisedDataset


def get_xy(dataloader):
    X, y = [], []
    for bx, by in dataloader:
        X.append(bx.numpy())
        y.append(by.numpy())
    return np.concatenate(X), np.concatenate(y)


def compute_auroc(clf, X_in, X_ood):
    p_in  = clf.predict_proba(X_in)
    p_ood = clf.predict_proba(X_ood)
    ent_in  = np.sum(p_in  * np.log(p_in  + 1e-10), axis=1)
    ent_ood = np.sum(p_ood * np.log(p_ood + 1e-10), axis=1)
    scores = np.concatenate([ent_in, ent_ood])
    labels = np.concatenate([np.zeros(len(X_in)), np.ones(len(X_ood))])
    return roc_auc_score(labels, scores)


def main(model_num, num_classes, epochs):
    from pytorch_tabnet.tab_model import TabNetClassifier

    binary = (num_classes == 2)

    X_wine,   _ = get_xy(createSklearnDataloader(load_wine(),          [0, 1, 2]))
    X_iris,   _ = get_xy(createSklearnDataloader(load_iris(),          [0, 1, 2]))
    X_cancer, _ = get_xy(createSklearnDataloader(load_breast_cancer(), [0, 1, 2]))

    test_accs, aurocs = [], []

    for run in tqdm(range(model_num), desc="Runs"):
        seed = run + 1
        _, train_loader, test_loader, _ = load_D1(seed, './datasets', binary=binary)

        X_train, y_train = get_xy(train_loader)
        X_test,  y_test  = get_xy(test_loader)

        noised_loader = DataLoader(GaussianNoisedDataset(train_loader.dataset),
                                   batch_size=1000, shuffle=False)
        X_noised, _ = get_xy(noised_loader)

        clf = TabNetClassifier(verbose=0, seed=seed)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=['accuracy'],
            max_epochs=epochs,
            patience=10,
            batch_size=16,
        )

        test_accs.append(np.mean(clf.predict(X_test) == y_test))
        aurocs.append([
            compute_auroc(clf, X_test, X_wine),
            compute_auroc(clf, X_test, X_iris),
            compute_auroc(clf, X_test, X_cancer),
            compute_auroc(clf, X_test, X_noised),
        ])

    print(f"TestAcc: {np.mean(test_accs):.3f} std {np.std(test_accs):.3f}")
    aurocs = np.array(aurocs)
    for i in range(4):
        print(f"AUROC {i+1}: {np.mean(aurocs[:, i]):.3f} std {np.std(aurocs[:, i]):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_num",   type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--epochs",      type=int, default=200)
    args = parser.parse_args()
    main(args.model_num, args.num_classes, args.epochs)
