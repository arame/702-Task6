import torch
import torchvision
from torchvision import transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from PIL import Image
from settings import Settings
from deviceGpu import DeviceGpu

class CKDataset(Dataset):

    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms

        assert len(images) == len(labels), 'Length of data and label should be same'
        self.length = len(images)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.images[idx]
        target = int(self.labels[idx])
        img = np.reshape(img, [100, 100,1]) # Reshape for convolutional

        target = int(self.labels[idx])
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target