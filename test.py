import torch
import torch.nn as nn
from Network import BaselineNet, cellsCountDistance, trainNet, testNet, BaselineNet_2
from ImageProcessing import compareTorchImages
from CellsDataset import CellsDataset, ToTensor, IndexSampler
from torchvision import transforms, utils
from torch.utils.data import DataLoader



# See if CUDA is available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Networks
net = BaselineNet_2().to(device)

#Dataset & Dataloaders
cellsDataset = CellsDataset(shuffle=True, root_dir = 'CellsDataset/', transform = ToTensor())
cellsDataset_n =CellsDataset(shuffle=True, root_dir= 'CellsDataset_n/', transform= ToTensor(), new=True)
num_samples = 200
split_size = 50
batch_size = 5
test_idxs = [range(s, s+split_size) for s in range(0, num_samples, split_size)]
train_idxs = [list(set(range(1, num_samples)) - set(idxs)) for idxs in test_idxs]
test_loaders = [DataLoader(cellsDataset, batch_size=batch_size, sampler=IndexSampler(idxs)) for idxs in test_idxs]
train_loaders = [DataLoader(cellsDataset, batch_size=batch_size, sampler=IndexSampler(idxs)) for idxs in train_idxs]

# 4-FOLD CROSS


partialPerformance = 0.0
for i in range(1):
    print('\nCrossing n.%d'%(i+1))
    trainNet(net, train_loaders[i], test_loaders[i], plot=False, device=device)
    split_performance = testNet(net, test_loaders[i], device)
    partialPerformance += split_performance
    print(f'MSE for split {i}: {split_performance}')
    if i == 3:
        print('## Final Performance ## :', partialPerformance/4)



