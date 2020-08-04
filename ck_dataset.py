import torch
from torch.utils.data import Dataset
import numpy as np

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