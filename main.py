import os,sys
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
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
from to_numpy import ToNumpy
from neural_net import ANN

def main():
    DeviceGpu.get()
    Settings.start()
    with open((Settings.pathInputFile), "rb") as fh:
        data = pickle.load(fh)
    print(data.keys())
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(data['training_data'][0][200].reshape((100, 100)), cmap="gray")
    fig.savefig(Settings.pathOutput + "firstImage.png", bbox_inches='tight', dpi=150)
    X_train = data['training_data'][0]
    y_train =data['training_data'][1]
    X_val = data['validation_data'][0]
    y_val =data['validation_data'][1]
    X_test= data['test_data'][0]
    y_test = data['test_data'][1]
    X= np.append(X_train, X_val, axis = 0)
    X= np.append(X, X_test, axis = 0)
    y = np.append(y_train, y_val, axis = 0)
    y= np.append(y, y_test, axis = 0)

    # Verification of the number of images and labels
    print("Labels ",len(y))
    print("Images ",len(X))

    # Data augmentation to the training data will be undertaken - T.RandomHorizontalFlip

    transform_train = T.Compose([T.ToPILImage(), T.RandomHorizontalFlip(0.5), ToNumpy(), T.ToTensor()])
    # Only tensor transformations will be done to the test and validation datasets
    transform = T.Compose([T.ToTensor()])
    # dataset and defined transformations
    train_dataset = CKDataset(X, y, transforms = transform_train)
    test_dataset = CKDataset(X, y, transforms= transform)
    # split the dataset in train, validation and test datasets using torch random permutations
    indices = torch.randperm(len(train_dataset)).tolist()
    idx= round(len(indices)*0.70)
    train_idx = round(len(indices)*0.85)
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:idx])
    val_dataset =   torch.utils.data.Subset(test_dataset, indices[idx:train_idx])
    test_dataset = torch.utils.data.Subset(test_dataset, indices[train_idx:])
    # loss funcion- cross entropy-softmax
    criterion = nn.CrossEntropyLoss()

    # Printing the neural network model - move the model to the available device
    ann_model = ANN(batch_norm = Settings.batch_norm, is_training= True ).to(DeviceGpu.device)
    # Move the model to the available device
    ann_model.to(DeviceGpu.device)

    print(ann_model)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Settings.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=Settings.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Settings.batch_size, shuffle=False)

    optimizer = optim(ann_model)
    #Defining learning scheduler -step

    #lr_schedulerst = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size= Settings.step_size, gamma=Settings.gamma)

    #Learning scheduler plateu
    lr_schedulerpl = ReduceLROnPlateau(optimizer=optimizer, mode='max',factor= Settings.factor, 
                                   patience=Settings.patience, verbose= True)

    since = time.time()

    best_acc = 0.0

    train_loss_log= []
    train_corrects_log = []
    val_loss_log= []
    val_corrects_log = []
    test_loss_log= []
    test_corrects_log = []

    for epoch in range(Settings.num_epochs):
        epoch_strlen = len(str(Settings.num_epochs)) 
        #getting the loss and accurracy from the training, validation and test datasets
        train_loss, train_acc = train(ann_model, train_loader, criterion, optimizer)

        val_loss, val_acc = test_val(ann_model, val_loader, criterion)
        test_loss, test_acc = test_val(ann_model, test_loader, criterion)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch= int(epoch + 1)
        #     best_model = copy.deepcopy(model.state_dict())

        lr_schedulerpl.step(test_acc)    
        sys.stderr.write('\r%0*d/%d | Train / Val/ Test loss.: %.3f / %.3f / %.3f '' | Train/Val/Test Acc.: %.3f%%/ %.3f%%/ %.3f%% ' 
                    % (epoch_strlen, epoch+1,Settings.num_epochs, train_loss, val_loss, test_loss, 
                    train_acc*100, val_acc*100, test_acc*100))
        print('| lr.:{:5f}'.format((optimizer.param_groups[0]['lr'])))

        train_loss_log.append(train_loss)
        train_corrects_log.append(train_acc)
        val_loss_log.append(val_loss)
        val_corrects_log.append(val_acc)
        test_loss_log.append(test_loss)
        test_corrects_log.append(test_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:3f}'.format(best_acc*100))
    print('Best test epoch: {:2d}'.format(best_epoch))
    #Visualising the cost vs epoc
    fig = plt.figure(figsize=(10, 10))  
    plt.plot(train_loss_log, label='training loss')
    plt.plot(val_loss_log, label='validation loss')
    plt.plot(test_loss_log, label='test loss')
    plt.legend()
    fig.savefig(Settings.pathOutput + "epoch_cost.png", bbox_inches='tight', dpi=150)

    fig = plt.figure(figsize=(10, 10)) 
    plt.plot(train_corrects_log, label='training accuracy')
    plt.plot(val_corrects_log, label='val accuracy')
    plt.plot(test_corrects_log, label='test accuracy')
    fig.savefig(Settings.pathOutput + "accuracy.png", bbox_inches='tight', dpi=150)

    try:
        torch.save(ann_model.state_dict(), Settings.pathSaveNet)
    except:
        print("Could not save model in ", Settings.pathSaveNet)
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # print images
    fig = plt.figure(figsize=(13, 7))
    grid = torchvision.utils.make_grid(images)
    imshow(grid)
    print('GroundTruth: ', ' '.join('%5s' % Settings.emotions[labels[j]] for j in range(Settings.batch_size)))

    try:
        ann_model.state_dict(torch.load(Settings.pathSaveNet)) # Change the path to your directory
    except:
        print("Unable to save model in the folder", Settings.pathSaveNet)

    outputs =ann_model(images.to(DeviceGpu.device))

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % Settings.emotions[predicted[j]] for j in range(Settings.batch_size)))

    class_correct = list(0. for i in range(len(Settings.emotions)))
    class_total = list(0. for i in range(len(Settings.emotions)))
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(DeviceGpu.device)
            labels = labels.to(DeviceGpu.device)

            outputs = ann_model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).squeeze()
            if correct.ndim == 0:
                continue
            for i in range(len(test_loader.dataset)):
                if i < len(labels):
                    label = labels[i].item()
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1
    for i in range(len(Settings.emotions)):
        print('Accuracy of %5s : %.2f %%' % (
            Settings.emotions[i], 100 * class_correct[i] / class_total[i]))

    # ----------THE END--------------------------------------------------------------
    Settings.end()

def imshow(img):
    fig = plt.figure(figsize=(10, 10))
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    fig.savefig(Settings.pathOutput + "epoch_cost.png", bbox_inches='tight', dpi=150)

# Defining SGD and Adam optimizers

def optim(model):
    if Settings.optimizer == "Adam":
        opts= torch.optim.Adam(model.parameters(), lr= Settings.learning_rate, weight_decay= Settings.weight_decay)
    if Settings.optimizer =="SGD":
        opts = torch.optim.SGD(model.parameters(), lr= Settings.learning_rate, momentum=Settings.momentum,
                    weight_decay= Settings.weight_decay)
    return opts

def train(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct_sum  = 0, 0
    for idx, (inputs, labels) in enumerate(loader):
        # data pixels and labels to available device
        inputs, labels = inputs.to(DeviceGpu.device), labels.to(DeviceGpu.device)
        # set the parameter gradients to zero
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss_sum += loss.item()*inputs.size(0)
        # propagate the loss backward
        loss.backward()
        # update the gradients
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        correct_sum += torch.sum(preds == labels) 
    #Statistics 
    train_loss = loss_sum/len(loader.dataset)
    train_acc = correct_sum.float()/len(loader.dataset)

    return train_loss, train_acc


def test_val(model, loader, criterion):
    model.eval()
    test_loss_sum, correct_sum  = 0, 0
    with torch.no_grad():
        loss_sum, correct_sum , total = 0, 0, 0
        for idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(DeviceGpu.device), labels.to(DeviceGpu.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss_sum +=  loss.item()*inputs.size(0)
            _, preds = torch.max(outputs, 1) 
            correct_sum += torch.sum(preds == labels)   
        #Statistics 
        loss = loss_sum/len(loader.dataset)
        acc = correct_sum.float()/len(loader.dataset)
        
    return loss, acc


if __name__ == '__main__':    
    main()