import argparse
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.load_datasets import *
from models.FastKANmodel import *
from train import *


def main(architecture, grids, learning_rate, use_base_update, epochs, weight_decay, model_num, dataset):
    architecture = list(map(int, architecture))
    numClasses = architecture[-1]

    trainLoader, testLoader, *falseloaders = loadAllDataloaders('./datasets', numClasses == 2, dataset=dataset)
    lossFunction = nn.CrossEntropyLoss()

    trainAccs, trainLosses, testAccs, testLosses, aurocs = [], [], [], [], []

    for _ in range(model_num):
        model = FastKAN(architecture, num_grids=grids, use_base_update=use_base_update).cuda()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

        trainAcc, trainLoss, testAcc, testLoss, auroc = networkTrain(
            'kan', model, optimizer, scheduler,
            lossFunction, trainLoader, testLoader, falseloaders, numClasses, None, epochs
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
    save_results('FastKAN', dataset,
             {'architecture': architecture, 'learning_rate': learning_rate, 'epochs': epochs, 'model_num': model_num},
             trainAccs, trainLosses, testAccs, testLosses, aurocs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture',  nargs='+', default=[3, 32, 3])
    parser.add_argument('--grids',         type=int,   default=4)
    parser.add_argument('--use_base_update',           type=lambda x: x.lower() != 'false', default=True)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--model_num',     type=int,   default=5)
    parser.add_argument('--dataset', type=str, default='ambrosia', choices=['ambrosia', 'heart'])
    main(**vars(parser.parse_args()))
