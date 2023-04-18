import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class SVDDNet(nn.Module):
    def __init__(self):
        super(SVDDNet, self).__init__()
        self.rep_dim = 1536
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

        self.fc1 = nn.Linear(256 * 6 * 32, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.elu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.elu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.elu(self.bn2d3(x)))
        x = self.conv4(x)
        x = self.pool(F.elu(self.bn2d4(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class VAENet(nn.Module):
    def __init__(self):
        super(VAENet, self).__init__()
        self.rep_dim = 1024
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

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

        self.fc1 = nn.Linear(256 * 6 * 32, 1536, bias=False)

        self.fc21 = nn.Linear(1536, self.rep_dim, bias=False)
        self.fc22 = nn.Linear(1536, self.rep_dim, bias=False)
        #self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04,affine=False)

        self.fc3 = nn.Linear(self.rep_dim, 1536, bias=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(
            int(1536 / (6 * 32)), 256, 5, bias=False, padding=2)
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

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_(
        ) if torch.cuda.is_available() else torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        x = self.conv1(x)
        unpooled_shape_1 = x.size()
        x, indices_1 = self.pool(F.elu(self.bn2d1(x)))
        x = self.conv2(x)
        unpooled_shape_2 = x.size()
        x, indices_2 = self.pool(F.elu(self.bn2d2(x)))
        x = self.conv3(x)
        unpooled_shape_3 = x.size()
        x, indices_3 = self.pool(F.elu(self.bn2d3(x)))
        x = self.conv4(x)
        unpooled_shape_4 = x.size()
        x, indices_4 = self.pool(F.elu(self.bn2d4(x)))
        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        z = self.reparameterize(mu, logvar)
        x = F.elu(self.fc3(z))

        x = x.view(x.size(0), int(1536 / (6 * 32)), 6, 32)
        x = F.elu(x)
        x = self.deconv1(x)
        x = self.unpool(F.elu(self.bn2d5(x)), indices_4,
                        output_size=unpooled_shape_4)
        x = self.deconv2(x)
        x = self.unpool(F.elu(self.bn2d6(x)), indices_3,
                        output_size=unpooled_shape_3)
        x = self.deconv3(x)
        x = self.unpool(F.elu(self.bn2d7(x)), indices_2,
                        output_size=unpooled_shape_2)
        x = self.deconv4(x)
        x = self.unpool(F.elu(self.bn2d8(x)), indices_1,
                        output_size=unpooled_shape_1)
        x = self.deconv5(x)
        x = torch.sigmoid(x)
        return x, mu, logvar
