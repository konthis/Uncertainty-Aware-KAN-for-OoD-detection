import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from datasets.load_datasets import loadAllDataloaders


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
    auroc = roc_auc_score(labels, scores)
    return max(auroc, 1.0 - auroc)


def main(model_num, num_classes, dataset):
    binary = (num_classes == 2)
    train_loader, test_loader, *false_loaders = loadAllDataloaders('./datasets', binary, dataset=dataset)
    X_train, y_train = get_xy(train_loader)
    X_test,  y_test  = get_xy(test_loader)
    X_oods = [get_xy(fl)[0] for fl in false_loaders]

    n_classes   = len(np.unique(y_train))
    eval_metric = 'logloss' if n_classes == 2 else 'mlogloss'

    test_accs, aurocs = [], []
    for run in tqdm(range(model_num), desc="Runs"):
        clf = XGBClassifier(
            n_estimators=1000,
            max_depth=3,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            reg_lambda=2.0,
            eval_metric=eval_metric,
            early_stopping_rounds=40,
            random_state=run + 1,
            verbosity=0,
        )
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        test_accs.append(np.mean(clf.predict(X_test) == y_test))
        aurocs.append([compute_auroc(clf, X_test, X_ood) for X_ood in X_oods])

    print(f"TestAcc: {np.mean(test_accs):.3f} std {np.std(test_accs):.3f}")
    aurocs = np.array(aurocs)
    for i in range(len(X_oods)):
        print(f"AUROC {i+1}: {np.mean(aurocs[:, i]):.3f} std {np.std(aurocs[:, i]):.3f}")
    from utils.save_results import save_results
    n = len(test_accs)
    save_results('XGBoost', dataset,
                 {'model_num': model_num},
                 [0]*n, [0]*n, test_accs, [0]*n, aurocs)
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_num", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument('--dataset',   type=str, default='ambrosia', choices=['ambrosia', 'heart'])
    main(**vars(parser.parse_args()))
