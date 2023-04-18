import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log

epsilon = 1e-06
log_eps = log(epsilon)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc31 = nn.Linear(32, 1)
        self.fc32 = nn.Linear(32, 1)
        self.fc33 = nn.Linear(32, 1)
        self.fc34 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        y1b = self.fc31(x)
        y1b = torch.clamp(y1b, min=log_eps, max=-log_eps)
        y1b = torch.log(torch.exp(y1b)+1.0) + 1.0

        y1a = self.fc32(x)
        y1a = torch.clamp(y1a, min=log_eps, max=-log_eps)
        y1a = torch.log(torch.exp(y1a)+1.0) + 1.0

        y1 = torch.clamp(y1a+y1b, min=epsilon)
        y1 = -5+10*(y1b/y1)

        y2b = self.fc33(x)
        y2b = torch.clamp(y2b, min=log_eps, max=-log_eps)
        y2b = torch.log(torch.exp(y2b)+1.0) + 1.0

        y2a = self.fc34(x)
        y2a = torch.clamp(y2a, min=log_eps, max=-log_eps)
        y2a = torch.log(torch.exp(y2a)+1.0) + 1.0

        y2 = torch.clamp(y2a+y2b, min=epsilon)
        y2 = 0+0.154333*(y2b/y2)

        return y1, y2


class ConstrinedLinear(nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight/(self.weight.norm()+1e-7))


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.rep_dim = 4

        # regressor network
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc31 = nn.Linear(32, 1)
        self.fc32 = nn.Linear(32, 1)
        self.fc33 = nn.Linear(32, 1)
        self.fc34 = nn.Linear(32, 1)

        # latent generator
        self.fc4 = ConstrinedLinear(2, self.rep_dim)

        self.fc5 = nn.Linear(8, 6, bias=False)
        self.fc61 = nn.Linear(6, self.rep_dim, bias=False)
        self.fc62 = nn.Linear(6, self.rep_dim, bias=False)
        self.fc7 = nn.Linear(4, 6, bias=False)
        self.fc8 = nn.Linear(6, 8, bias=False)

        self.parameters = [self.fc4.weight, self.fc4.bias.data,
                           self.fc5.weight,
                           self.fc61.weight,
                           self.fc62.weight,
                           self.fc7.weight,
                           self.fc8.weight]

    def init_regressor_weights(self, network):
        self.fc1.weight.data = network.fc1.weight.data
        self.fc1.bias.data = network.fc1.bias.data

        self.fc2.weight.data = network.fc2.weight.data
        self.fc2.bias.data = network.fc2.bias.data

        self.fc31.weight.data = network.fc31.weight.data
        self.fc31.bias.data = network.fc31.bias.data

        self.fc32.weight.data = network.fc32.weight.data
        self.fc32.bias.data = network.fc32.bias.data

        self.fc33.weight.data = network.fc33.weight.data
        self.fc33.bias.data = network.fc33.bias.data

        self.fc34.weight.data = network.fc34.weight.data
        self.fc34.bias.data = network.fc34.bias.data

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def forward(self, input):

        x = input*torch.tensor([100.0, 100.0, 120.0, 10.0,
                                120.0, 120.0, 60.0, 60.0]).to(self.device)
        #print 'x'
        #print x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        y1b = self.fc31(x)
        y1b = torch.clamp(y1b, min=log_eps, max=-log_eps)
        y1b = torch.log(torch.exp(y1b)+1.0) + 1.0

        y1a = self.fc32(x)
        y1a = torch.clamp(y1a, min=log_eps, max=-log_eps)
        y1a = torch.log(torch.exp(y1a)+1.0) + 1.0

        y1 = torch.clamp(y1a+y1b, min=epsilon)
        y1 = -5+10*(y1b/y1)
        #print 'y1'
        #print y1

        y2b = self.fc33(x)
        y2b = torch.clamp(y2b, min=log_eps, max=-log_eps)
        y2b = torch.log(torch.exp(y2b)+1.0) + 1.0

        y2a = self.fc34(x)
        y2a = torch.clamp(y2a, min=log_eps, max=-log_eps)
        y2a = torch.log(torch.exp(y2a)+1.0) + 1.0

        y2 = torch.clamp(y2a+y2b, min=epsilon)
        #print 'y21'
        #print y2
        y2 = 0+0.154333*(y2b/y2)
        #print 'y22'
        #print y2

        try:
            y = torch.cat([y1, y2], 1)
        except:
            y = torch.cat([y1, y2], -1)

        pz_mu = self.fc4(y)

        x = F.elu(self.fc5(input))
        mu = self.fc61(x)
        logvar = self.fc62(x)
        z = self.reparameterize(mu, logvar)
        x = F.elu(self.fc7(z))
        x = self.fc8(x)
        return x, mu, logvar, pz_mu, y
