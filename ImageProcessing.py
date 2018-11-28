from skimage import io, transform, filters
from skimage.util import view_as_blocks
from skimage import color
import os
import plotly.plotly as py
import plotly.graph_objs as go
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan)
import warnings

def conv2grayFrR(image):
    grayImage = image[:,:,0]
    return grayImage

def conv2grayFrB(image):
    grayImage = image[:,:,2]
    return grayImage

def gaussianConv(image, sigma):
    return filters.gaussian(image, sigma, multichannel=False)

def scaleHeatmap(image, newSize):  #WARNING: this method work well only in the new tuple is a divider for old size
    if isinstance(newSize, tuple):
        blockedImage = view_as_blocks(image, (image.shape[0]//newSize[0], image.shape[1]//newSize[1]))
        scaledImage = np.zeros(newSize)
        for i in range(blockedImage.shape[1]):
            for j in range(blockedImage.shape[0]):
                scaledImage[i,j] = np.array(np.sum(blockedImage[i,j]))
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



#TEST

'''image = io.imread(os.path.abspath('CellsDataset/001dots.png'))

grayImage = conv2grayFrR(image)
gaussImage = gaussianConv(grayImage, 7)
scaledImage = scaleHeatmap(gaussImage, (64,64))

print(grayImage.sum()/255)
print(gaussImage.sum())
print(scaledImage.sum())

plt.imshow(gaussImage)
plt.show()'''








