from torch import nn
import torch.nn.functional as F
from models.CNNmodel import *

class MLP(nn.Module):
    def __init__(self, architecture,activationFunction):
        super().__init__()
        self.architecture = architecture
        inputSize = architecture[0]
        layers = []
        for l in architecture[1:]:   
            layers.append(nn.Linear(inputSize,l))
            layers.append(activationFunction)
            inputSize = l
        
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)

class SoftmaxModel(nn.Module):
    def __init__(self, architecture,activationFunction):
        super().__init__()
        self.model = MLP(architecture,activationFunction)

    def forward(self,x):
        return F.softmax(self.model(x),dim=1)

class CNNSoftmax(nn.Module):
    def __init__(self,cnnout, mlparch, activationFunction):
        super().__init__()
        self.cnn = CNN(cnnout)
        self.mlp = MLP(architecture=mlparch,activationFunction=activationFunction)
    def forward(self,x):
        return self.mlp(self.cnn.calc_features(x))