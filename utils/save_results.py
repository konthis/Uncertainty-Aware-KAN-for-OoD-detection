import os
import csv
import numpy as np
from datetime import datetime


def save_results(model_name, dataset, hyperparams: dict, 
                 trainAccs, trainLosses, testAccs, testLosses, aurocs, weighted_aurocs=None):

    out_dir = os.path.join('./results', model_name, f"{dataset}")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    aurocs      = np.array(aurocs)       # [runs, n_ood]
    n_ood       = aurocs.shape[1]

    # --- CSV ---
    csv_path = os.path.join(out_dir, 'results.csv')
    headers  = ['timestamp', 'train_acc', 'train_acc_std', 'test_acc', 'test_acc_std',
                 'train_loss', 'train_loss_std', 'test_loss', 'test_loss_std']
    headers += [f'AUROC{i+1}_mean' for i in range(n_ood)]
    headers += [f'AUROC{i+1}_std'  for i in range(n_ood)]

    if weighted_aurocs is not None:
        weighted_aurocs = np.array(weighted_aurocs)
        headers += [f'W_AUROC{i+1}_mean' for i in range(n_ood)]
        headers += [f'W_AUROC{i+1}_std'  for i in range(n_ood)]

    row = {
        'timestamp':       timestamp,
        'train_acc':       f"{np.mean(trainAccs):.4f}",
        'train_acc_std':   f"{np.std(trainAccs):.4f}",
        'test_acc':        f"{np.mean(testAccs):.4f}",
        'test_acc_std':    f"{np.std(testAccs):.4f}",
        'train_loss':      f"{np.mean(trainLosses):.4f}",
        'train_loss_std':  f"{np.std(trainLosses):.4f}",
        'test_loss':       f"{np.mean(testLosses):.4f}",
        'test_loss_std':   f"{np.std(testLosses):.4f}",
    }
    for i in range(n_ood):
        row[f'AUROC{i+1}_mean'] = f"{np.mean(aurocs[:, i]):.4f}"
        row[f'AUROC{i+1}_std']  = f"{np.std(aurocs[:, i]):.4f}"
    if weighted_aurocs is not None:
        for i in range(n_ood):
            row[f'W_AUROC{i+1}_mean'] = f"{np.mean(weighted_aurocs[:, i]):.4f}"
            row[f'W_AUROC{i+1}_std']  = f"{np.std(weighted_aurocs[:, i]):.4f}"

    write_header = not os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    # --- Hyperparams txt ---
    with open(os.path.join(out_dir, f'hyperparams_{timestamp}.txt'), 'w') as f:
        f.write(f"Model:   {model_name}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Time:    {timestamp}\n\n")
        f.write("Hyperparameters:\n")
        for k, v in hyperparams.items():
            f.write(f"  {k}: {v}\n")

    print(f"Saved to {out_dir}")
