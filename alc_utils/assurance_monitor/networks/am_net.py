import torch
import torch.nn as nn
import torch.nn.functional as F


class SVDDNet(nn.Module):
    def __init__(self):
        super(SVDDNet, self).__init__()
        self.rep_dim = 4
        self.fc1 = nn.Linear(8, 6, bias=False)
        self.fc2 = nn.Linear(6, self.rep_dim, bias=False)

    def forward(self, x):
        #x= x.float()
        #print('in fwd', x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.rep_dim = 4
        self.fc1 = nn.Linear(8, 6, bias=False)
        self.fc21 = nn.Linear(6, self.rep_dim, bias=False)
        self.fc22 = nn.Linear(6, self.rep_dim, bias=False)
        self.fc3 = nn.Linear(4, 6, bias=False)
        self.fc4 = nn.Linear(6, 8, bias=False)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def forward(self, x):
        x = F.elu(self.fc1(x))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        z = self.reparameterize(mu, logvar)
        x = F.elu(self.fc3(z))
        x = self.fc4(x)
        return x, mu, logvar
