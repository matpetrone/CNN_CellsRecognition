import torch
import torch.nn as nn
from Network import BaselineNet, cellsCountDistance, trainNet, testNet
from ImageProcessing import compareTorchImages
from CellsDataset import CellsDataset, ToTensor, ChunkSampler
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


net = BaselineNet().to(device)

cellsDataset = CellsDataset(shuffle = True, root_dir = 'CellsDataset/', transform = ToTensor())
batch_size = 50
dataLoader_1 = DataLoader(cellsDataset, batch_size=4, shuffle=False, num_workers=4, sampler=ChunkSampler(batch_size, 0))
dataLoader_2 = DataLoader(cellsDataset, batch_size=4, shuffle=False, num_workers=4, sampler=ChunkSampler(batch_size, len(cellsDataset)//4))
dataLoader_3 = DataLoader(cellsDataset, batch_size=4, shuffle=False, num_workers=4, sampler=ChunkSampler(batch_size, len(cellsDataset)//2))
dataLoader_4 = DataLoader(cellsDataset, batch_size=4, shuffle=False, num_workers=4, sampler=ChunkSampler(batch_size, (len(cellsDataset)//2)+(len(cellsDataset)//4)))
dataloaders = [dataLoader_1, dataLoader_2, dataLoader_3, dataLoader_4]

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
    dataloaders_train = dataloaders[1:]
    if i > 0:
        dataloaders_train = dataloaders[0:i] + dataloaders[(i+1):]
    trainNet(net, dataloaders_train)
    partialPerformance += testNet(net, dataloaders[i])


