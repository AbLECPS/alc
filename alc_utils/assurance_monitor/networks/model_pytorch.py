import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log

epsilon = 1e-06
log_eps = log(epsilon)

# Mods
# This one is the network model.
# It has been updated to accept a parameter lrp during initialization
# If lrp is True, then it will output 4 values during network evaluation


class Network(nn.Module):
    def __init__(self, lrp=0):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc31 = nn.Linear(32, 1)
        self.fc32 = nn.Linear(32, 1)
        self.fc33 = nn.Linear(32, 1)
        self.fc34 = nn.Linear(32, 1)
        self.lrp = lrp

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        if (self.lrp == 0 or self.lrp == 2):
            y1b = self.fc31(x)
            y1b = torch.clamp(y1b, min=log_eps, max=-log_eps)
            y1b = torch.log(torch.exp(y1b)+1.0) + 1.0
            if (self.lrp == 2):
                return y1b

        if (self.lrp == 0 or self.lrp == 1):
            y1a = self.fc32(x)
            y1a = torch.clamp(y1a, min=log_eps, max=-log_eps)
            y1a = torch.log(torch.exp(y1a)+1.0) + 1.0
            if (self.lrp == 1):
                return y1a

        if (self.lrp == 0 or self.lrp == 4):
            y2b = self.fc33(x)
            y2b = torch.clamp(y2b, min=log_eps, max=-log_eps)
            y2b = torch.log(torch.exp(y2b)+1.0) + 1.0
            if (self.lrp == 4):
                return y2b

        if (self.lrp == 0 or self.lrp == 3):
            y2a = self.fc34(x)
            y2a = torch.clamp(y2a, min=log_eps, max=-log_eps)
            y2a = torch.log(torch.exp(y2a)+1.0) + 1.0
            if (self.lrp == 3):
                return y2a

        y1 = torch.clamp(y1a+y1b, min=epsilon)
        y1 = -5+10*(y1b/y1)

        y2 = torch.clamp(y2a+y2b, min=epsilon)
        y2 = 0+0.154333*(y2b/y2)

        return y1, y2
