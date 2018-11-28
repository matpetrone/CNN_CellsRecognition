from CellsDataset import CellsDataset
from skimage import io, transform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from CellsDataset import CellsDataset, ToTensor
from torch.utils.data import DataLoader
from ImageProcessing import conv2grayFrB


class BaselineNet(torch.nn.Module):

   def __init__(self):
       super(BaselineNet, self).__init__()
       self.conv1 = nn.Conv2d(1,32,kernel_size=3, padding=1)
       self.maxPool1 = nn.MaxPool2d(3,stride=2)
       self.conv2 = nn.Conv2d(32,64,kernel_size=3, padding=1)
       self.maxPool2 = nn.MaxPool2d(3, stride=2)
       self.conv3 = nn.Conv2d(64,1,kernel_size=3, padding=1)

   def forward(self, x):
       in_size = x.shape[0]
       x = F.relu(self.maxPool1(self.conv1(x)))
       x = F.relu(self.maxPool2(self.conv2(x)))
       x = self.conv3(x)
       x = x.view(-1, self.num_flat_features(x))
       return x

   def num_flat_features(self, x):
       size = x.size()[1:]  # all dimensions except the batch dimension
       num_features = 1
       for s in size:
           num_features *= s
       return num_features

#TEST
dataset = CellsDataset(csv_file = 'cells_landmarks.csv', root_dir = 'CellsDataset/', transform=ToTensor())
sample = dataset[0]['image']
sample = sample[2,:,:] #greyScale
print(sample.shape)
net = BaselineNet()
print(net(sample))