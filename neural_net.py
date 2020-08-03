import os,sys
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.optim as optim 
import pickle
import time
import datetime
import copy
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from settings import Settings
from deviceGpu import DeviceGpu
from ck_dataset import CKDataset

class ANN(nn.Module):
    def __init__(self, batch_norm = False, is_training = True): #p1,p2,
        super(ANN, self).__init__()
      
        self.training = is_training
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, 
                               kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, 
                               kernel_size=5, stride=1)
        if batch_norm:
            self.conv2_bn = nn.BatchNorm2d(50)
            self.fc1_bn = nn.BatchNorm1d(200) # out features
        self.conv2_drop = nn.Dropout2d(p=Settings.drop_prob1) #p1=0.4
        self.fc1 = nn.Linear(in_features=24200, out_features= 200)  #22*22*50
        self.fc1_drop = nn.Dropout2d(p=Settings.drop_prob2) #p2=05
        self.fc2 = nn.Linear(in_features=200, out_features= 8) #8 number of classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        if Settings.batch_norm:
            x = self.conv2_bn(self.conv2(x))
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = self.conv2_drop(x)
            x = x.view(-1, 24200)
            x= self.fc1(x)
            x= self.fc1_bn(x)
            x= F.relu(x)
            x = self.fc1_drop(x)

        else:
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = self.conv2_drop(x)
            x = x.view(-1, 24200)
            x = F.relu(self.fc1(x))
            x = self.fc1_drop(x)         
        x = self.fc2(x)
      
        return x 
