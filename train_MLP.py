import argparse
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.load_datasets import *
from models.MLPmodel import *
from train import *


def main(architecture, learning_rate, epochs, model_num, dataset):
    architecture = list(map(int, architecture))
    numClasses = architecture[-1]

    trainLoader, testLoader, *falseloaders = loadAllDataloaders('./datasets', numClasses == 2, dataset=dataset)

    lossFunction = nn.CrossEntropyLoss()

    trainAccs, trainLosses, testAccs, testLosses, aurocs = [], [], [], [], []

    for _ in range(model_num):
        model = SoftmaxModel(architecture, activationFunction=nn.ReLU()).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

        trainAcc, trainLoss, testAcc, testLoss, auroc = networkTrain(
            'mlp', model, optimizer, scheduler,
            lossFunction, trainLoader, testLoader, falseloaders, numClasses, None, epochs
        )
        # i like this ; operator
        trainAccs.append(trainAcc); trainLosses.append(trainLoss)
        testAccs.append(testAcc);   testLosses.append(testLoss)
        aurocs.append(auroc)

    print(f"TrainAcc: {np.mean(trainAccs):.3f} std {np.std(trainAccs):.3f},  TrainLoss: {np.mean(trainLosses):.3f} std {np.std(trainLosses):.3f}")
    print(f"TestAcc:  {np.mean(testAccs):.3f} std {np.std(testAccs):.3f},  TestLoss:  {np.mean(testLosses):.3f} std {np.std(testLosses):.3f}")
    aurocs_mean, aurocs_std = np.mean(aurocs, axis=0), np.std(aurocs, axis=0)
    for i, (m, s) in enumerate(zip(aurocs_mean, aurocs_std)):
        print(f"AUROC {i+1}: {m:.3f} std {s:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', nargs='+', default=[3, 32, 16, 3])
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--model_num',     type=int,   default=5)
    parser.add_argument('--dataset', type=str, default='ambrosia', choices=['ambrosia', 'heart'])
    main(**vars(parser.parse_args()))
