from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,n_output=8,kernel_size=4,maxpool_kernel_size = 4, padding = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(1,n_output,kernel_size, padding)
        self.maxp = nn.MaxPool2d(maxpool_kernel_size)
        self.bn1 = nn.BatchNorm2d(n_output)

    def calc_features(self, x):
        x = F.relu(self.bn1(self.maxp(self.conv1(x))))
        x = x.flatten(1)
        return x
