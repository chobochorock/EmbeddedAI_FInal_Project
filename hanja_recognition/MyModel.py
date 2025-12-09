import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MyDataset import MyDataset

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        in_channels     = 1
        out_channels    = 1
        self.conv       = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.flatten    = nn.Flatten()
        self.linear     = nn.Linear(50*50, 10)

        self.criterion  = nn.CrossEntropyLoss()
        self.optimizer  = torch.optim.SGD(self.parameters(), lr = 0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
    
    def optimize(self, pred, label) :
        self.optimizer.zero_grad()
        for param in self.optimizer.param_groups:
            # print(param['lr'])
            param['lr'] *= 0.999
        loss = self.criterion(pred, label)
        loss.backward()
        self.optimizer.step()

    # =========================================================================
    # do not modify the following codes
    # =========================================================================
    def save(self, path='model.pth'):
        device = torch.device('cpu')
        self.to(device)
        torch.save(self.state_dict(), path)

    def load(self, path='model.pth'):
        device = torch.device('cpu')
        self.to(device)
        self.load_state_dict(torch.load(path))

    def size(self):
        size = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return size
    
    def print(self):
        print(self.state_dict())
    # =========================================================================