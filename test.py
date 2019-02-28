import torch
import torch.nn as nn
from Network import BaselineNet, cellsCountDistance, trainNet, testNet
from ImageProcessing import compareTorchImages
from CellsDataset import CellsDataset, ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


net = BaselineNet().to(device)

cellsDataset = CellsDataset(csv_file = 'cells_landmarks.csv', root_dir = 'CellsDataset/', transform = ToTensor())
testloader = DataLoader(cellsDataset, batch_size=4, shuffle=False, num_workers=4)
trainloader = DataLoader(cellsDataset, batch_size=4, shuffle=True, num_workers=4)


#exemple cycle to train-test
trainNet(net, trainloader)
testNet(net,testloader)