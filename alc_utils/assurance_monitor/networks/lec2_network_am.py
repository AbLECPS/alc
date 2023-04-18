import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict


class SVDD(nn.Module):
    def __init__(self):
        super(SVDD, self).__init__()
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


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

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


class VisualBackPropNet(nn.Module):
    def __init__(self):
        super(VisualBackPropNet, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            1, 1, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(
            1, 1, kernel_size=3, stride=2, padding=(0, 1))

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
        self.conv = [self.conv1, self.conv1,
                     self.conv2, self.conv1, self.conv1]

    def normalization(self, tensor):
        omin = tensor.min(2, keepdim=True)[0].min(3, keepdim=True)[0].mul(-1)
        omax = tensor.max(2, keepdim=True)[0].max(3, keepdim=True)[0].add(omin)
        tensor = torch.add(tensor, omin.expand(tensor.size(
            0), tensor.size(1), tensor.size(2), tensor.size(3)))
        tensor = torch.div(tensor, omax.expand(tensor.size(
            0), tensor.size(1), tensor.size(2), tensor.size(3)))
        return tensor

    def forward(self, maps, size):
        #print ('in backprop forward')
        map = maps[-1]
        # for i in range(len(self.conv)-1):
        #    x2 = maps[-i-2]
        #    print('x2 size',x2.size())
        for i in range(len(self.conv)-1):
            x2 = maps[-i-2]
            x1 = self.conv[i](map, output_size=x2.size())
            #print 'x1 ', x1.shape
            #print 'x2 ', x2.shape
            map = torch.mul(x1, x2)
            map = self.normalization(map)

        #print('output size',size)
        output = self.conv[-1](map, output_size=size)
        return output


class segnet(nn.Module):
    def __init__(self, n_classes=21, in_channels=3, is_unpooling=True):
        super(segnet, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)

        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, n_classes)

    def forward(self, inputs):
        features = []

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        features.append(down1.sum(1, keepdim=True))
        down2, indices_2, unpool_shape2 = self.down2(down1)
        features.append(down2.sum(1, keepdim=True))
        down3, indices_3, unpool_shape3 = self.down3(down2)
        features.append(down3.sum(1, keepdim=True))
        down4, indices_4, unpool_shape4 = self.down4(down3)
        features.append(down4.sum(1, keepdim=True))
        down5, indices_5, unpool_shape5 = self.down5(down4)
        features.append(down5.sum(1, keepdim=True))

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        return up1, features

    def init_vgg16_params(self, vgg16):
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data


class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices,
                              output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices,
                              output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation,)

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state

    """
    new_state_dict = OrderedDict()
    for k, v in list(state_dict.items()):
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_model(model_file_name):
    loaded_model_state = torch.load(model_file_name)
    model = segnet(n_classes=4)
    vgg16 = models.vgg16(pretrained=False)
    model.init_vgg16_params(vgg16)
    state = convert_state_dict(loaded_model_state["model_state"])
    model.load_state_dict(state)
    return model
