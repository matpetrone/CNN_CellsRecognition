import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from CellsDataset import CellsDataset, ToTensor
from torch.utils.data import DataLoader
from ImageProcessing import visualizeTorchImage, compareTorchImages, randomCrop, convertTorchToNp, convertNptoTorch
import torch.optim as optim
from tensorboardX import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaselineNet(torch.nn.Module):

   def __init__(self):
       super(BaselineNet, self).__init__()
       self.conv1 = nn.Conv2d(1,32,kernel_size=3, padding=1)
       self.maxPool1 = nn.MaxPool2d(2,stride=2)
       self.conv2 = nn.Conv2d(32,64,kernel_size=3, padding=1)
       self.maxPool2 = nn.MaxPool2d(2, stride=2)
       self.conv3 = nn.Conv2d(64,1,kernel_size=3, padding=1)

   def forward(self, x):
       x = F.relu(self.maxPool1(self.conv1(x)))
       x = F.relu(self.maxPool2(self.conv2(x)))
       x = self.conv3(x)
       s = x.sum((1,2,3))
       return (x, s)

def rmse(predictions, target):                         #RMSE between two lists
    return np.sqrt(((predictions- target)**2).mean())

def cellsCountDistance(image1, image2):                #Distance between number of cells in two pics
    prediction = torch.sum(image1)
    target = torch.sum(image2)
    prediction = prediction.detach().numpy()
    target = target.detach().numpy()
    return rmse(prediction, target)

def meanCellsCount(prediction, target, j):
    partialMSE = 0.0
    for i in range(len(prediction)):
        partialMSE += cellsCountDistance(prediction[i], target[i])
    partialMSE = partialMSE/len(prediction)
    print('Mean Squared Error batch n.%d:'%j, partialMSE)
    return partialMSE

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.reset_parameters()

def resetNetParameters(net):
    net.apply(weights_init)
    print('Net has been initialized')

def subImages(images, n_crop = 2):
    if torch.is_tensor(images[0]):
        images = convertTorchToNp(images)
    rankedImages = []
    for i in range(len(images)):
        subsetImgs = randomCrop(images[i], n_crop)
        for j in range(n_crop):
            rankedImages.append(subsetImgs[j])
    return np.array(rankedImages)

def rankingImages(sum_batch, n_crop, n_origImg):
    partialLoss = torch.autograd.Variable(torch.tensor([0.0]))
    m = 2
    for i in range(n_origImg):
        for j in range(n_crop):
            partialLoss += torch.clamp(sum_batch[n_origImg + j + (i*n_crop)] - sum_batch[i] + m, min=0)
    return partialLoss


#TRAINING

def trainNet(net, dataloaders):
    criterion = nn.MSELoss()
    optimizerAdam = optim.Adam(net.parameters(), lr=0.0001)
    resetNetParameters(net)
    for epoch in range(80):  #loop over the dataset multiple times
        running_loss = 0.0
        rn_loss_MSE = 0.0
        rn_loss_R = 0.0
        for dataloader in dataloaders:
            for i, data in enumerate(dataloader, 0):
                # get the inputs
                inputs = data['image']
                labels = data['landmarks']

                #Ranking Loss
                n_crops = 2
                crops = subImages(inputs, n_crops) #create an array that contains in each pos. subimages randomly cropped
                crops = convertNptoTorch(crops, resize = True)
                inputs = torch.cat((inputs,crops),0)

                # zero the parameter gradients
                optimizerAdam.zero_grad()

                # forward + backward + optimize
                (outputs, s) = net(inputs)
                loss_MSE = criterion(outputs[0:labels.shape[0]], labels)
                loss_R = rankingImages(s, n_crops, labels.shape[0])
                loss = loss_MSE + loss_R
                loss.backward()
                optimizerAdam.step()

                # print statistics
                rn_loss_MSE += loss_MSE.item()
                rn_loss_R += loss_R
                running_loss += loss.item()

                if i % 10 == 9:    # print every 20 mini-batches
                    print('[%d, %5d] loss: %.7f, loss_MSE: %.7f, loss_R: %.7f' %
                          (epoch + 1, i + 1, running_loss / 10, rn_loss_MSE / 10, rn_loss_R / 10))
                    running_loss = 0.0
                    rn_loss_MSE = 0.0
                    rn_loss_R = 0.0
                    compareTorchImages(outputs[0], labels[0])   #show picture comparison

    print('Finished Training')


#TESTING

def testNet(net, dataloader):
    distanceMSE = 0.0
    for j, data in enumerate(dataloader, 0):
        inputs = data['image']
        labels = data['landmarks']

        (outputs, _) = net(inputs)

        distanceMSE += meanCellsCount(outputs,labels, j+1)
    print('Finished Testing')
    distanceMSE /= len(dataloader)
    return distanceMSE