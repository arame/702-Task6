import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import Settings

class ANN(nn.Module):
    def __init__(self, batch_norm = False, is_training = True): #p1,p2,
        super(ANN, self).__init__()
      
        self.training = is_training
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
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
        if Settings.batch_norm:
            x= self.fc1_bn(x)
        x= F.relu(x)
        x = self.fc1_drop(x)
        x = self.fc2(x)
      
        return x 
