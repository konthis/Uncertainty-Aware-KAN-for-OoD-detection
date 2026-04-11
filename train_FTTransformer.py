import argparse
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.load_datasets import loadAllDataloaders
from train import networkTrain


class FTTransformerWrapper(nn.Module):
    """Wraps rtdl FTTransformer to accept a single tensor (no categorical features)."""
    def __init__(self, n_features, n_classes):
        super().__init__()
        import rtdl
        self.model = rtdl.FTTransformer.make_default(
            n_num_features=n_features,
            cat_cardinalities=[],
            d_out=n_classes,
        )

    def forward(self, x):
        return self.model(x, None)


def main(learning_rate, epochs, model_num, num_classes, dataset):
    binary = (num_classes == 2)
    # use a dummy architecture to get numClasses — infer from data
    train_loader, test_loader, *falseloaders = loadAllDataloaders('./datasets', binary, dataset=dataset)

    # infer dimensions from first batch
    x_sample, y_sample = next(iter(train_loader))
    n_features = x_sample.shape[1]
    n_classes  = int(train_loader.dataset.tensors[1].max().item()) + 1

    lossFunction = nn.CrossEntropyLoss()

    trainAccs, trainLosses, testAccs, testLosses, aurocs = [], [], [], [], []

    for _ in range(model_num):
        model = FTTransformerWrapper(n_features, n_classes).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

        trainAcc, trainLoss, testAcc, testLoss, auroc = networkTrain(
            'ft_transformer', model, optimizer, scheduler,
            lossFunction, train_loader, test_loader, falseloaders, n_classes, None, epochs
        )
        trainAccs.append(trainAcc); trainLosses.append(trainLoss)
        testAccs.append(testAcc);   testLosses.append(testLoss)
        aurocs.append(auroc)

    print(f"TrainAcc: {np.mean(trainAccs):.3f} std {np.std(trainAccs):.3f},  TrainLoss: {np.mean(trainLosses):.3f} std {np.std(trainLosses):.3f}")
    print(f"TestAcc:  {np.mean(testAccs):.3f} std {np.std(testAccs):.3f},  TestLoss:  {np.mean(testLosses):.3f} std {np.std(testLosses):.3f}")
    aurocs_mean, aurocs_std = np.mean(aurocs, axis=0), np.std(aurocs, axis=0)
    for i, (m, s) in enumerate(zip(aurocs_mean, aurocs_std)):
        print(f"AUROC {i+1}: {m:.3f} std {s:.3f}")
    
    from utils.save_results import save_results
    save_results('FTTransformer', dataset,
             {'learning_rate': learning_rate, 'epochs': epochs, 'model_num': model_num},
             trainAccs, trainLosses, testAccs, testLosses, aurocs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--model_num',     type=int,   default=5)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument('--dataset',       type=str,   default='ambrosia', choices=['ambrosia', 'heart'])
    main(**vars(parser.parse_args()))
