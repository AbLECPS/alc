# FIXME: This skeleton needs work.
import torch
import torch.nn as nn
import torch.nn.functional as F


# Some of the assurance monitor algorithms train their own Neural Net model, and thus require an architecture to train.
# The architecture of this NN can have a significant impact on the overall performance of the assurance algorithm.
# This skeleton provides an example architecture definition intended for the "DAVE2" network architecture and
# compatible with the "SVDD" assurance algorithm.
class SVDD(nn.Module):
    def __init__(self):
        super(SVDD, self).__init__()
        # self.rep_dim = 1536
        # self.pool = nn.MaxPool2d(2, 2)
        # # Encoder
        # self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        # nn.init.xavier_uniform_(self.conv1.weight)
        # self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        # self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        # nn.init.xavier_uniform_(self.conv2.weight)
        # self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        # self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        # nn.init.xavier_uniform_(self.conv3.weight)
        # self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        # self.conv4 = nn.Conv2d(128, 256, 5, bias=False, padding=2)
        # nn.init.xavier_uniform_(self.conv4.weight)
        # self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        # self.fc1 = nn.Linear(256 * 4 * 12, self.rep_dim, bias=False)
        pass

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.pool(F.elu(self.bn2d1(x)))
        # x = self.conv2(x)
        # x = self.pool(F.elu(self.bn2d2(x)))
        # x = self.conv3(x)
        # x = self.pool(F.elu(self.bn2d3(x)))
        # x = self.conv4(x)
        # x = self.pool(F.elu(self.bn2d4(x)))
        # x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        # return x
        pass
