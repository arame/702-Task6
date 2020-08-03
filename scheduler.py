import torch.optim.lr_scheduler as lr
from settings import Settings

class Scheduler:
    @staticmethod
    def getAdjustLearningRate(optimizer):
        if Settings.isPlateau:
            return lr.ReduceLROnPlateau(optimizer=optimizer, mode='max',factor= Settings.factor, patience=Settings.patience, verbose= True)
        else:
            return lr.StepLR(optimizer=optimizer, step_size= Settings.step_size, gamma=Settings.gamma)
