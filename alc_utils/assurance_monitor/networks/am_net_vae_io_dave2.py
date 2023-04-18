import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log

epsilon = 1e-06
log_eps = log(epsilon)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),

        )
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=1152, out_features=1164),
            nn.ELU(),
            nn.Linear(in_features=1164, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, data):
        output = self.conv_layers(data)
        output = output.view(output.size(0), -1)
        output = self.dense_layers(output)

        return output


class ConstrinedLinear(nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight/(self.weight.norm()+1e-7))


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.rep_dim = 1024

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),

        )
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=1152, out_features=1164),
            nn.ELU(),
            nn.Linear(in_features=1164, out_features=100),
            nn.ELU(),

            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1)
        )
        # latent generator
        self.fc4 = ConstrinedLinear(1, self.rep_dim)

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.conv4 = nn.Conv2d(128, 256, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(256 * 4 * 12, 1536, bias=False)

        self.fc21 = nn.Linear(1536, self.rep_dim, bias=False)
        self.fc22 = nn.Linear(1536, self.rep_dim, bias=False)
        #self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04,affine=False)

        self.fc3 = nn.Linear(self.rep_dim, 1536, bias=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(
            int(1536 / (4 * 12)), 256, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight)
        self.bn2d5 = nn.BatchNorm2d(256, eps=1e-04, affine=False)

        self.deconv2 = nn.ConvTranspose2d(256, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight)
        self.bn2d6 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.deconv3 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight)
        self.bn2d7 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv4 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight)
        self.bn2d8 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.deconv5 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv5.weight)

        self.parameters = []

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def init_regressor_weights(self, network):
        for i in [0, 2, 4, 6, 8]:
            self.conv_layers[i].weights.data = network.conv_layers[i].weights.data
            self.conv_layers[i].bias.data = network.conv_layers[i].bias.data

        for i in [0, 2, 4, 6, 8]:
            self.dense_layers[i].weights.data = network.dense_layers[i].weights.data
            self.dense_layers[i].bias.data = network.dense_layers[i].bias.data

    def forward(self, x):
        output = self.conv_layers(x)
        output = output.view(x.size(0), -1)
        output = self.dense_layers(x)

        pz_mu = self.fc4(output)

        x = self.conv1(x)
        x = self.pool(F.elu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.elu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.elu(self.bn2d3(x)))
        x = self.conv4(x)
        x = self.pool(F.elu(self.bn2d4(x)))
        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        z = self.reparameterize(mu, logvar)
        x = F.elu(self.fc3(z))

        x = x.view(x.size(0), int(1536 / (4 * 12)), 4, 12)
        x = F.elu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.elu(self.bn2d5(x)), size=[8, 25])
        x = self.deconv2(x)
        x = F.interpolate(F.elu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.elu(self.bn2d7(x)), size=[33, 100])
        x = self.deconv4(x)
        x = F.interpolate(F.elu(self.bn2d8(x)), scale_factor=2)
        x = self.deconv5(x)
        x = torch.sigmoid(x)
        return x, mu, logvar, pz_mu, output
