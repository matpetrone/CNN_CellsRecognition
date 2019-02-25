from __future__ import print_function, division
import os
import torch
import csv
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from ImageProcessing import createHeatMap64
np.set_printoptions(threshold=np.nan)


class CellsDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file, error_bad_lines=False)
        self.root_dir = root_dir
        self.transform = transform
        self.rowCount = self.landmarks_frame.apply(lambda x: x.count(), axis=1)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_path)
        dots_path = os.path.join(self.root_dir, convertFileName(self.landmarks_frame.iloc[idx, 0]))
        landmarks = io.imread(dots_path)
        landmarks = createHeatMap64(landmarks)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated



class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}

def convertFileName(string1):
    string = list(string1)
    string[3] = 'd'
    string[4] = 'o'
    string[5] = 't'
    string[6] = 's'
    newString = "".join(string)
    return newString



#TEST

cells_dataset = CellsDataset(csv_file = 'cells_landmarks.csv', root_dir = 'CellsDataset/', transform = ToTensor())


for i in range(len(cells_dataset)):
    sample = cells_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break

dataloader = DataLoader(cells_dataset, batch_size=4,shuffle=True, num_workers=4)

def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
'''plt.imshow(cells_dataset[1]['landmarks'])
plt.show()
'''












