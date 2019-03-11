import torch
import torch.nn as nn
from Network import BaselineNet, cellsCountDistance, trainNet, testNet
from ImageProcessing import compareTorchImages
from CellsDataset import CellsDataset, ToTensor, IndexSampler
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# See if CUDA is available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = BaselineNet().to(device)

cellsDataset = CellsDataset(shuffle=True, root_dir = 'CellsDataset/', transform = ToTensor())
num_samples = 200
split_size = 50
batch_size = 5
test_idxs = [range(s, s+split_size) for s in range(0, num_samples, split_size)]
train_idxs = [list(set(range(1, num_samples)) - set(idxs)) for idxs in test_idxs]
test_loaders = [DataLoader(cellsDataset, batch_size=batch_size, sampler=IndexSampler(idxs)) for idxs in test_idxs]
train_loaders = [DataLoader(cellsDataset, batch_size=batch_size, sampler=IndexSampler(idxs)) for idxs in train_idxs]

# 4-FOLD CROSS VALIDATION
'''partialPerformance = 0.0
for i in range(len(dataloaders)):
    if i==0:
        print('## Starting 4-Cross Training ##')
    print('\nCrossing n.%d'%(i+1))
    dataloaders_train = dataloaders[1:]
    if i > 0:
        dataloaders_train = dataloaders[0:i] + dataloaders[(i+1):]
    trainNet(net, dataloaders_train)
    partialPerformance += testNet(net, dataloaders[i])
totalPerformance = partialPerformance / len(dataloaders)
print('Final Performance:', totalPerformance)'''

#1st-PASS
partialPerformance = 0.0
for i in range(1):
    print('\nCrossing n.%d'%(i+1))
    trainNet(net, train_loaders[i], test_loaders[i], plot=False, device=device)
    split_performance = testNet(net, test_loaders[i], device)
    partialPerformance += split_performance
    print(f'MSE for split {i}: {split_performance}')


