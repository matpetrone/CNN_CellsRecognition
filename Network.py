import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from CellsDataset import CellsDataset, ToTensor
from torch.utils.data import DataLoader
from ImageProcessing import visualizeTorchImage, compareTorchImages, randomCrop, convertTorchToNp, convertNptoTorch_arr, randomFlip
import torch.optim as optim
from adamw import AdamW
from tensorboardX import SummaryWriter
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#NETWORKS

class BaselineNet(torch.nn.Module):
   def __init__(self):
       super(BaselineNet, self).__init__()
       self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
       self.maxPool1 = nn.MaxPool2d(2,stride=2)
       self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
       self.maxPool2 = nn.MaxPool2d(2, stride=2)
       self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

   def forward(self, x):
       x = F.relu(self.maxPool1(self.conv1(x)))
       x = F.relu(self.maxPool2(self.conv2(x)))
       x = self.conv3(x)
       s = x.sum((1,2,3))
       return (x, s)


class BaselineNet_drop(torch.nn.Module):
   def __init__(self):
       super(BaselineNet_drop, self).__init__()
       self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
       self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
       self.conv_drop1 = nn.Dropout2d()
       self.maxPool1 = nn.MaxPool2d(2,stride=2)
       self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
       self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
       self.conv_drop2 = nn.Dropout2d()
       self.maxPool2 = nn.MaxPool2d(2, stride=2)
       self.conv5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
       self.conv3_drop = nn.Dropout2d()

   def forward(self, x):
       x = F.relu(self.maxPool1(self.conv2(self.conv_drop1(self.conv1(x)))))
       x = F.relu(self.maxPool2(self.conv4(self.conv_drop2(self.conv3(x)))))
       x = self.conv3(x)
       s = x.sum((1,2,3))
       return (x, s)

class BaselineNet_2(torch.nn.Module):
   def __init__(self):
       super(BaselineNet_2, self).__init__()
       self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
       self.conv2 = nn.Conv2d(32,32, kernel_size=3, padding=1)
       self.maxPool1 = nn.MaxPool2d(2,stride=2)
       self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
       self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
       self.maxPool2 = nn.MaxPool2d(2, stride=2)
       self.conv5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

   def forward(self, x):
       x = F.relu(self.maxPool1(self.conv2(self.conv1(x))))
       x = F.relu(self.maxPool2(self.conv4(self.conv3(x))))
       x = self.conv5(x)
       s = x.sum((1,2,3))
       return (x, s)


# TRAINING
def trainNet(net, dataloader, test_loader=None, plot=False, device='cpu'):
    net.train()
    criterion = nn.MSELoss()
    optimizerAdam = optim.Adam(net.parameters(), weight_decay=1e-5, lr=0.0001)
    optimizerAdamW = AdamW(net.parameters(),lr=0.0001)
    resetNetParameters(net)
    for epoch in range(300):  #loop over the dataset multiple times
        running_loss = 0.0
        rn_loss_MSE = 0.0
        rn_loss_R = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs
            inputs = data['image']
            labels = data['landmarks']

            #Ranking Loss
            n_crops = 3
            ranking_weight= 0.0001
            crops = subImages(inputs, n_crops) #create an array that contains in each pos. subimages randomly cropped
            crops = convertNptoTorch_arr(crops, resize = True)
            inputs = torch.cat((inputs,crops),0)

            #Randomly Flip inputs
            for j in range(labels.shape[0]):
                (inputs[j], labels[j]) = randomFlip(inputs[j], labels[j])
            labels = labels.to(device)
            for j in range(labels.shape[0], inputs.shape[0]):
                (inputs[j], _) = randomFlip(inputs[j])

            # zero the parameter gradients
            optimizerAdamW.zero_grad()

            # forward + backward + optimize
            (outputs, s) = net(inputs.to(device))
            loss_MSE = criterion(outputs[0:labels.shape[0]], labels)
            loss_R = rankingImages(s, n_crops, labels.shape[0], device)
            loss = loss_MSE + ranking_weight*loss_R
            loss.backward()
            optimizerAdamW.step()


            # print statistics
            rn_loss_MSE += loss_MSE.item()
            rn_loss_R += loss_R
            running_loss += loss.item()

            if i % 20 == 19:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.7f, loss_MSE: %.7f, loss_R: %.7f' %
                      (epoch + 1, i + 1, running_loss / 20, rn_loss_MSE / 20, rn_loss_R / 20))
                running_loss = 0.0
                rn_loss_MSE = 0.0
                rn_loss_R = 0.0
                if plot:
                    compareTorchImages(outputs[0], labels[0])   #show picture comparison

        if epoch % 5 == 4:
            print('MSE on test: {}'.format(testNet(net, test_loader, device)))
    print('Finished Training')


#TESTING

def testNet(net, dataloader, device='cpu'):
    net.eval()
    distanceMSE = torch.Tensor([0.0]).to(device)
    for j, data in enumerate(dataloader, 0):
        inputs = data['image'].to(device)
        labels = data['landmarks']
        (outputs, _) = net(inputs)
        distanceMSE += meanCellsCount(outputs.cpu(), labels, j+1)

    distanceMSE /= len(dataloader)
    return distanceMSE



#RANKING METHODS

def subImages(images, n_crop = 2):
    if torch.is_tensor(images[0]):
        images = convertTorchToNp(images)
    rankedImages = []
    for i in range(len(images)):
        subsetImgs = randomCrop(images[i], n_crop)
        for j in range(n_crop):
            rankedImages.append(subsetImgs[j])
    return np.array(rankedImages)

def rankingImages(sum_batch, n_crop, n_origImg, device='cpu'):
    partialLoss = torch.autograd.Variable(torch.tensor([0.0])).to(device)
    m = 1
    for i in range(n_origImg):
        for j in range(n_crop):
            partialLoss += torch.clamp(sum_batch[n_origImg + j + (i*n_crop)] - sum_batch[i] + m, min=0)
    return partialLoss



#ACCURACY TEST & NET OP

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