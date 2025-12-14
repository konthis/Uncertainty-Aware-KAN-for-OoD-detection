import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.load_datasets import *
from models.ProposedKANmodel import *
from utils.functions import *
from train import *

def main():
    lossFunction = ProposedLoss(lamda=0.25,tau=0.04)
    baseActivation = ActivationFunctions(gamma=4.).RBF_SiLU

    trainLoader,testLoader,falseLoader1,falseLoader2,falseLoader3,_= loadAllDataloaders('./datasets',True)
    falseloaders = [falseLoader1,falseLoader2,falseLoader3]
    _,trainloader,testloader,_ = load_D1(2,'./datasets',binary=True)
    noisedtrainloader = load_noisedDataset(trainloader.dataset)
    noisedtestloader = load_noisedDataset(testloader.dataset)
    trainAccs = []
    trainLosses = []
    testAccs = []
    testLosses = []
    aurocs = []

    for _ in range(1):
        model = ProposedKAN([3,32,2],num_grids=4,base_activation=baseActivation,denominator=1).cuda()
        denomParam = [p for name, p in model.named_parameters() if 'denominator' in name]
        othersParam = [p for name, p in model.named_parameters() if 'denominator' not in name]
        optimizer = torch.optim.SGD([{"params":othersParam},{"params":denomParam,"lr":1e-3}],
                                    lr = 1e-3,weight_decay=1e-4,momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

        trainAcc,trainLoss,testAcc,testLoss,auroc = networkTrain('kan',model,optimizer,scheduler,
                                                                 lossFunction,trainLoader,noisedtestloader,
                                                                 falseloaders,
                                                                 2,0,50)
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
    main()