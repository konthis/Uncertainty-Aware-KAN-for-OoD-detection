import torch
from torch import nn
import torch.nn.functional as F
from models.CNNmodel import *

class DUQ(nn.Module):
    def __init__(self, inputDim,outFeatureDim,centroidDim,outputDim,std,initSigma):
        super().__init__()
        self.fc1 = nn.Linear(inputDim, outFeatureDim)
        ## DUQ weight vector, Parameter so the optimizer and backprop get it into consideration
        ## size = centroidDim x noClasses x featureDim(prev output)
        self.W = nn.Parameter(
            torch.normal(torch.zeros(centroidDim, outputDim, outFeatureDim),std=1)
        )

        # length scale
        self.sigma = initSigma
        # momentum
        self.gamma = 0.999

        self.std = std #  standard div of m
        #centroids are calculated as e_ct = m_ct/n_ct, c=class t = minibatch
        # register buffers = parameters that dont return with parameters() call, so it wont calc the derivs for backprop
        self.register_buffer("n",torch.ones(outputDim))
        self.register_buffer('m', torch.normal(torch.zeros(centroidDim, outputDim), std = std))


        # for pirnting
        self.architecture = [inputDim, outFeatureDim, centroidDim, outputDim]

    def forwardFeatures(self,x):
        x = F.relu(self.fc1(x))
        return x

    def embeddingLayer(self, x):
        #  last weight layer, on DUQ part
        # simple matrix mul
        # x size = batchSize x outFeatureDim
        # z size = batchSize x centroidDim x noclasses
        z = torch.einsum("ij,mnj->imn", x, self.W)
        return z

    def update_embeddings(self,x,y):
        z = self.embeddingLayer(self.forwardFeatures(x))
        self.n = torch.max(self.gamma * self.n + (1 - self.gamma) * y.sum(0), torch.ones_like(self.n)) # IF 0 SAMPLES OF A CLASS FOUND, SET IT TO 1
        self.m = self.gamma * self.m + (1 - self.gamma) * torch.einsum("ijk,ik->jk",z,y) # einsum with onehot enc y, to activate on correct class

    def calcDistanceLayer(self, z):
        # centroids
        e = self.m / self.n
        diff = z - e 
        distances = (-(diff**2)).mean(1).div(2 * self.sigma**2).exp()
        return distances

    def forward(self,x):
        # features
        x = self.forwardFeatures(x)
        # embed
        z = self.embeddingLayer(x)
        # distances/certainty
        ypred = self.calcDistanceLayer(z)
        return ypred

class CNNDUQ(nn.Module):
    def __init__(self,cnnout, inputDim,outFeatureDim,centroidDim,outputDim,std,initSigma):
        super().__init__()
        self.cnn = CNN(cnnout)
        self.DUQ= DUQ(inputDim,outFeatureDim,centroidDim,outputDim,std,initSigma)
    def forward(self,x):
        return self.DUQ(self.cnn.calc_features(x))
    def update_embeddings(self,x,y):
        self.DUQ.update_embeddings(self.cnn.calc_features(x),y)