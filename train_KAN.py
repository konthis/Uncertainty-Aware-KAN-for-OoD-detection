import argparse

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.load_datasets import *
from models.KANmodel import *
from train import *


def main(architecture,
         grids,
         learning_rate,
         epochs,
         gamma,
         weight_decay,
         model_num):

    architecture = list(map(int, architecture)) 
    numClasses = architecture[-1]

    lossFunction = nn.CrossEntropyLoss()

    trainLoader,testLoader,falseLoader1,falseLoader2,falseLoader3,falseLoader4 = loadAllDataloaders('./datasets',True if numClasses==2 else False)
    falseloaders = [falseLoader1,falseLoader2,falseLoader3,falseLoader4]
    trainAccs = []
    trainLosses = []
    testAccs = []
    testLosses = []
    aurocs = []

    for _ in range(model_num):
        model = KAN(width=architecture,grid=grids).cuda()

        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=gamma, weight_decay=weight_decay
        )
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

        trainAcc,trainLoss,testAcc,testLoss,auroc = networkTrain('kan',model,optimizer,scheduler,
                                                                 lossFunction,trainLoader,testLoader,falseloaders,
                                                                 numClasses,None,epochs)
        trainAccs.append(trainAcc)
        trainLosses.append(trainLoss)
        testAccs.append(testAcc)
        testLosses.append(testLoss)
        aurocs.append(auroc)
    print(f"TrainAcc:{np.mean(trainAccs):>.3f} std {np.std(trainAccs):>.3f}, TrainLoss:{np.mean(trainLosses):>.3f} std {np.std(trainLosses):>.3f}")
    print(f"TestAcc:{np.mean(testAccs):>.3f} std {np.std(testAccs):>.3f}, TrainLoss:{np.mean(testLosses):>.3f} std {np.std(testLosses):>.3f}")
    aurocsR = np.mean(aurocs,axis=0)
    aurocsStd = np.std(aurocs,axis=0)
    for i in range(len(aurocsR)):
        print(f"AUROC {i+1}: {aurocsR[i]:>.3f} std {aurocsStd[i]:>.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--architecture',
        nargs='+',
        default=[3,32,3],
        help='Input architecture',
    )
    parser.add_argument(
        "--grids",
        type=int,
        default=4,
        help="Grids (default: 4)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Epochs (default: 15)",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="Decay factor for exponential average (default: 0.9)",
    )

    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)"
    )

    parser.add_argument(
        "--model_num", 
        type=int, 
        default=5, 
        help="Number of models to train (default: 5)"
    )

    args = parser.parse_args()
    kwargs = vars(args)

    main(**kwargs)