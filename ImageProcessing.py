from skimage import io, transform, filters
from scipy.ndimage import gaussian_filter
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan)
import warnings
import torch
import random
import cv2

def conv2grayFrR(image):
    grayImage = image[:,:,0]
    return grayImage /255.0  #Normalize

def conv2grayFrB(image):
    grayImage = image[:,:,2]
    return grayImage.reshape(256,256,1) /255.0 #Normalize

def gaussianConv(image, sigma):
    return gaussian_filter(image, sigma)

def scaleHeatmap(image, newSize):  #WARNING: this method work well only if the new tuple is a divider for old size
    if isinstance(newSize, tuple):
        blockedImage = view_as_blocks(image, (image.shape[0]//newSize[0], image.shape[1]//newSize[1]))
        scaledImage = np.zeros(newSize)
        for i in range(blockedImage.shape[1]):
            for j in range(blockedImage.shape[0]):
                scaledImage[i,j] = np.array(np.sum(blockedImage[i,j]))
                scaledImage = scaledImage
        return scaledImage
    else:
        warnings.warn('WARNING: second input must be a tuple (n, m)')

def sparseImage(image):
    newImage = np.zeros(image.shape)
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            if image[i,j] != 0 :
                newImage[i,j] = 1
    return newImage

def createHeatMap(image, newSize, sigma = 7):
    heatMap = conv2grayFrR(image)
    heatMap1 = gaussianConv(heatMap, sigma)
    heatMap2 = scaleHeatmap(heatMap1, newSize)
    heatMap2 = heatMap2.reshape(newSize[0], newSize[1], 1)
    return heatMap2

def visualizeTorchImage(tensor, str =''):
    image = tensor.detach().numpy()
    image = image.transpose((1,2,0))
    image = image.reshape((image.shape[0], image.shape[1]))
    plt.imshow(image)
    plt.title(str)
    plt.show()

def convertTorchToNp(tensor):
    images = []
    if len(tensor.shape) > 3:  #if it's a batch
        for i in range(tensor.shape[0]):
            image = tensor[i].numpy()
            image = image.transpose((1,2,0))
            image = image.reshape((image.shape[0], image.shape[1]))
            images.append(image)
    else:
        image = tensor.numpy()
        image = image.transpose((1,2,0))
        image = image.reshape((image.shape[0], image.shape[1]))
        return image
    return np.array(images)

def convertNptoTorch_arr(images, resize = False):
    imgs = []
    for i in range(len(images)):
        img = images[i]
        if resize:
            img = resizeImage(img)
        img = img.reshape(img.shape[0], img.shape[1], 1)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        imgs.append(img)
    images = torch.cat((imgs),0)
    return images

def convertNptoTorch(image):
    if image is None:
        return None
    image = image.reshape(image.shape[0], image.shape[1], 1)
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    #image = image.unsqueeze(0)
    return image

def compareTorchImages(tensor1, tensor2):
    plt.subplot(1,2,1)
    image1 = tensor1.view(tensor1.shape[1],tensor1.shape[2], tensor1.shape[0])
    image1 = np.squeeze(image1.cpu().detach().numpy())
    image1 = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))
    plt.imshow(image1)
    plt.title('CNN Output')

    plt.subplot(1,2,2)
    image2 = tensor2.view(tensor2.shape[1], tensor2.shape[2], tensor2.shape[0])
    image2 = np.squeeze(image2.cpu().detach().numpy())
    image2 = (image2 - np.min(image2)) / (np.max(image2) - np.min(image2))
    plt.imshow(image2)
    plt.title('Landmark')

    plt.show()

def visualizeNpImage(img):
    plt.imshow(img)
    plt.show()

def randomCrop(image, n_crop = 1):
    if torch.is_tensor(image):
        image = convertTorchToNp(image) #WARNING: return a shaped np H x W, NOT H x W x 1
    h, w = image.shape[:2]
    cropImages = []
    output_size = 64
    for i in range(n_crop):
        output_size = np.min(np.random.randint(output_size,image.shape[0], size=6))
        new_h, new_w = (output_size, output_size)
        anchor = np.random.randint(new_h//2, h-(new_h//2))
        start_idx = anchor - (new_h//2)
        end_idx = anchor + (new_h//2)
        cropImg = image[start_idx:end_idx, start_idx:end_idx]
        cropImages.append(cropImg)
    return cropImages


def resizeImage(image, newSize=(256,256)):
    return cv2.resize(image, dsize=newSize, interpolation=cv2.INTER_CUBIC)

def randomFlip(image, label=None, p=0.25):
    lab = False
    if torch.is_tensor(image):
        image = convertTorchToNp(image)    #WARNING: return a shaped np H x W,  NOT  H x W x 1
        if torch.is_tensor(label):
            label = convertTorchToNp(label)
            lab = True
    nut = random.random()
    if nut > p:
        if nut < 0.5:
            if lab:
                label = np.flipud(label).copy()
            image = np.flipud(image).copy()
            return (convertNptoTorch(image), convertNptoTorch(label))
        elif nut < 0.75:
            if lab:
                label = np.fliplr(label).copy()
            image = np.fliplr(image).copy()
            return (convertNptoTorch(image), convertNptoTorch(label))
        elif nut < 1.0:
            if lab:
                label = np.flipud(label).copy()
                label = np.fliplr(label).copy()
            image = np.flipud(image).copy()
            image = np.fliplr(image).copy()
            return (convertNptoTorch(image), convertNptoTorch(label))
    return (convertNptoTorch(image), convertNptoTorch(label))

def flipUD(image):
    if torch.is_tensor(image):
        image = convertTorchToNp(image) #WARNING: return a shaped np H x W,  NOT  H x W x 1
    return convertNptoTorch(np.flipud(image).copy())

def flipLR(image):
    if torch.is_tensor(image):
        image = convertTorchToNp(image) #WARNING: return a shaped np H x W,  NOT  H x W x 1
    return convertNptoTorch(np.fliplr(image).copy())




