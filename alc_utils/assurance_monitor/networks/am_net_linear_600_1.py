import torch
import torch.nn as nn
import torch.nn.functional as F


class SVDDNet(nn.Module):
    def __init__(self):
        super(SVDDNet, self).__init__()
        self.rep_dim = 64
        self.h_dim = 256
        self.fce1 = nn.Linear(600, self.h_dim, bias=False)
        self.fce2 = nn.Linear(self.h_dim, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.fce1(x)
        x = F.relu(x)
        x = self.fce2(x)
        x = F.relu(x)
        return x


class VAENet(nn.Module):
    def __init__(self):
        super(VAENet, self).__init__()
        self.rep_dim = 64
        self.h_dim = 256
        self.fce1 = nn.Linear(600, self.h_dim, bias=False)
        self.fce2m = nn.Linear(self.h_dim, self.rep_dim, bias=False)
        self.fce2v = nn.Linear(self.h_dim, self.rep_dim, bias=False)
        self.fcd1 = nn.Linear(self.rep_dim, self.h_dim, bias=False)
        self.fcd2 = nn.Linear(self.h_dim, 600, bias=False)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def forward(self, x):
        x = F.elu(self.fce1(x))
        mu = self.fce2m(x)
        logvar = self.fce2v(x)
        z = self.reparameterize(mu, logvar)
        x = F.elu(self.fcd1(z))
        x = self.fcd2(x)
        return x, mu, logvar
