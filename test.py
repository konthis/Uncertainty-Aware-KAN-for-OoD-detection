
import torch
import torch.nn.functional as F
import numpy as np
from medmnist import PathMNIST,ChestMNIST,DermaMNIST,OCTMNIST
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

def one_hot_encode_label(label, num_classes):
    if label.ndim == 1: 
        return F.one_hot(label, num_classes=num_classes).type(torch.LongTensor).squeeze()
    else:
        return label.type(torch.LongTensor).squeeze()

def loadPathMNIST(targetDir):
    transform = transforms.Compose([
                transforms.Resize((28, 28)),  
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                transforms.Lambda(lambda x: torch.flatten(x))])

    mnist_train = PathMNIST(root=targetDir, split="train", download=True, transform=transform)
    print(one_hot_encode_label(torch.from_numpy(mnist_train[10][1]),9))
    #print(mnist_train[10].shape)
    #trainloader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    #testloader  = DataLoader(mnist_test, batch_size=64, shuffle=True)
    #return trainloader, testloader

loadPathMNIST('./datasets')