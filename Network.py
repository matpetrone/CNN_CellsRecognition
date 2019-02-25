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
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaselineNet(torch.nn.Module):

   def __init__(self):
       super(BaselineNet, self).__init__()
       self.conv1 = nn.Conv2d(1,32,kernel_size=3, padding=1)
       self.maxPool1 = nn.MaxPool2d(3,stride=2)
       self.conv2 = nn.Conv2d(32,64,kernel_size=3, padding=1)
       self.maxPool2 = nn.MaxPool2d(3, stride=2)
       self.conv3 = nn.Conv2d(64,1,kernel_size=3, padding=1)

   def forward(self, x):
       #in_size = x.shape[0]
       x = F.relu(self.maxPool1(self.conv1(x)))
       x = F.relu(self.maxPool2(self.conv2(x)))
       x = self.conv3(x)
       #x = x.view(in_size, -1)
       return x

   def num_flat_features(self, x):
       size = x.size()[1:]  # all dimensions except the batch dimension
       num_features = 1
       for s in size:
           num_features *= s
       return num_features


net = BaselineNet().to(device)


#TRAINING
cellsDataset = CellsDataset(csv_file = 'cells_landmarks.csv', root_dir = 'CellsDataset/', transform=ToTensor())
dataloader = DataLoader(cellsDataset, batch_size=4, shuffle=True, num_workers=4)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        inputs = data['image']
        labels = data['landmarks']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')



#TEST
'''sample = cellsDataset[0]['image']
sample = sample.float()
print(type(sample))
sample = sample[2,:,:] #greyScale
print(sample.shape)
sample = sample.unsqueeze(0).unsqueeze(1)

out = net(sample)
print(out)'''