from __future__ import print_function, division
import os
import torch
import csv
import pandas as pd
from skimage import io, transform
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils
from ImageProcessing import conv2grayFrB, createHeatMap,visualizeNpImage
np.set_printoptions(threshold=np.nan)


class CellsDataset(Dataset):

    def __init__(self, root_dir, shuffle=True, transform=None):
        if shuffle:
            csv_file = 'cells_landmarks_rand.csv'
        else:
            csv_file = 'cells_landmarks.csv'
        self.landmarks_frame = pd.read_csv(csv_file, error_bad_lines=False)
        self.root_dir = root_dir
        self.transform = transform
        self.rowCount = self.landmarks_frame.apply(lambda x: x.count(), axis=1)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_path)
        image = conv2grayFrB(image)
        dots_path = os.path.join(self.root_dir, convertFileName(self.landmarks_frame.iloc[idx, 0]))
        landmarks = io.imread(dots_path)
        landmarks = createHeatMap(landmarks, (64,64))
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        landmarks = landmarks.transpose ((2,0,1))
        newImg = torch.from_numpy(image)
        newLmks = torch.from_numpy(landmarks)
        return {'image': newImg.float(), 'landmarks': newLmks.float()}

def convertFileName(string1):
    string = list(string1)
    string[3] = 'd'
    string[4] = 'o'
    string[5] = 't'
    string[6] = 's'
    newString = "".join(string)
    return newString

class IndexSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, indexes):
        self._indexes = indexes

    def __iter__(self):
        return iter(np.random.permutation(self._indexes))

    def __len__(self):
        return len(self._indexes)
    

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(np.random.permutation(range(self.start, self.start + self.num_samples)))

    def __len__(self):
        return self.num_samples
