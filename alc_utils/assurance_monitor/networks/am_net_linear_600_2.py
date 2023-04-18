import torch
import torch.nn as nn
import torch.nn.functional as F


class SVDDNet(nn.Module):
    def __init__(self):
        super(SVDDNet, self).__init__()
        self.rep_dim = 32
        self.fce1 = nn.Linear(600, 512, bias=False)
        self.fce2 = nn.Linear(512, 256, bias=False)
        self.fce3 = nn.Linear(256,128, bias=False)
        self.fce4 = nn.Linear(128,64, bias=False)
        self.fce5 = nn.Linear(64, self.rep_dim, bias=False)

        

    def forward(self, x):
        x = self.fce1(x)
        x = F.relu(x)
        x = self.fce2(x)
        x = F.relu(x)
        x = self.fce3(x)
        x = F.relu(x)
        x = self.fce4(x)
        x = F.relu(x)
        x = self.fce5(x)
        x = F.relu(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.rep_dim = 32
        self.fce1 = nn.Linear(600, 512, bias=False)
        self.fce2 = nn.Linear(512, 256, bias=False)
        self.fce3 = nn.Linear(256,128, bias=False)
        self.fce4 = nn.Linear(128,64, bias=False)
        self.fce5m = nn.Linear(64, self.rep_dim, bias=False)
        self.fce5v = nn.Linear(64, self.rep_dim, bias=False)
        self.fcd1 = nn.Linear(self.rep_dim, 64, bias=False)
        self.fcd2 = nn.Linear(64, 128, bias=False)
        self.fcd3 = nn.Linear(128,256, bias=False)
        self.fcd4 = nn.Linear(256,512, bias=False)
        self.fcd5 = nn.Linear(512,600, bias=False)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def forward(self, x):
        x = self.fce1(x)
        x = F.relu(x)
        x = self.fce2(x)
        x = F.relu(x)
        x = self.fce3(x)
        x = F.relu(x)
        x = self.fce4(x)
        x = F.relu(x)
        mu = self.fce5m(x)
        logvar = self.fce5v(x)
        z = self.reparameterize(mu, logvar)
        x = F.elu(self.fcd1(z))
        x = F.elu(self.fcd2(x))
        x = F.elu(self.fcd3(x))
        x = F.elu(self.fcd4(x))
        x = self.fcd5(x)
        return x, mu, logvar
