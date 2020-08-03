import torch
from enum import Enum

class DeviceType(Enum):
    cuda = 1
    cpu = 2

class DeviceGpu:
    type = DeviceType.cpu
    device = None

    @staticmethod
    def get():
        if torch.cuda.is_available():
            DeviceGpu.type = DeviceType.cuda
            DeviceGpu.device = torch.device('cuda')
        else:
            DeviceGpu.type = DeviceType.cpu
            DeviceGpu.device = torch.device('cpu')

