import argparse
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.load_datasets import *
from models.DUQmodel import *
from train import *


def main(architecture, learning_rate, epochs, l_gradient_penalty, length_scale, model_num, dataset):
    architecture = list(map(int, architecture))
    numClasses = architecture[-1]
    centroid_size = architecture[-2]

    trainLoader, testLoader, *falseloaders = loadAllDataloaders('./datasets', numClasses == 2, dataset=dataset)

    lossFunction = nn.CrossEntropyLoss()

    trainAccs, trainLosses, testAccs, testLosses, aurocs = [], [], [], [], []

    for _ in range(model_num):
        model = DUQ(architecture[0], architecture[1], centroid_size, numClasses, 1e-2, length_scale).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

        trainAcc, trainLoss, testAcc, testLoss, auroc = networkTrain(
            'duq', model, optimizer, scheduler,
            lossFunction, trainLoader, testLoader, falseloaders, numClasses, l_gradient_penalty, epochs
        )
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
    parser.add_argument('--architecture',       nargs='+', default=[3, 32, 16, 3])
    parser.add_argument('--learning_rate',      type=float, default=0.1)
    parser.add_argument('--epochs',             type=int,   default=120)
    parser.add_argument('--l_gradient_penalty', type=float, default=0.25)
    parser.add_argument('--length_scale',       type=float, default=1.0)
    parser.add_argument('--model_num',          type=int,   default=5)
    parser.add_argument('--dataset', type=str, default='ambrosia', choices=['ambrosia', 'heart'])
    main(**vars(parser.parse_args()))
