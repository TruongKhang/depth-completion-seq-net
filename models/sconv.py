import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.utils import _pair, _quadruple

from base.base_model import BaseModel

from utils.util import retrieve_elements_from_indices


class SparseConv(nn.Module):
    # Convolution layer for sparse data
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(SparseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.if_bias = bias
        if self.if_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation)

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.pool.require_grad = False

    def forward(self, input):
        x, m = input
        mc = m.expand_as(x)
        x = x * mc
        x = self.conv(x)

        weights = torch.ones_like(self.conv.weight)
        mc = F.conv2d(mc, weights, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
        mc = torch.clamp(mc, min=1e-5)
        mc = 1. / mc
        x = x * mc
        if self.if_bias:
            x = x + self.bias.view(1, self.bias.size(0), 1, 1).expand_as(x)
        m = self.pool(m)

        return x, m


class SparseMaxPooling(nn.Module):
    def __init__(self, kernel_size=2):
        super(SparseMaxPooling, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x, m = input
        mc = m.expand_as(x)
        x = x * mc
        x = F.max_pool2d(x, self.kernel_size)
        m = F.max_pool2d(m, self.kernel_size)
        return x, m


class SparseUpsampling(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=None):
        super(SparseUpsampling, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.epsilon = 1e-20

    def forward(self, input):
        x, m = input
        mc = m.expand_as(x)
        x = x * mc
        x_up = F.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)
        m_up = F.interpolate(m, self.size, self.scale_factor, self.mode, self.align_corners)
        z = x_up / (m_up.expand_as(x_up) + self.epsilon)
        m_z = (m_up > 0).float()
        return z, m_z


class SparseConcatConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SparseConcatConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, input_x, input_y):
        x, m_x = input_x
        y, m_y = input_y
        z = torch.cat((x, y), 1)
        z = m_x*(1-m_y)*self.conv1(z) + (1-m_x)*m_y*self.conv2(z) + m_x*m_y*self.conv3(z)
        m_z = m_x*(1-m_y) + (1-m_x)*m_y + m_x*m_y
        return z, m_z


class SparseConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(SparseConvBlock, self).__init__()
        self.sparse_conv = SparseConv(in_channel, out_channel, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x1, m1 = self.sparse_conv(input)
        assert (m1.size(1) == 1)
        x1 = self.relu(x1)
        return x1, m1


class SparseConvNet(BaseModel):

    def __init__(self, in_channel=1, out_channel=1, kernels=(11, 7, 5, 3, 3), mid_channel=16):
        super(SparseConvNet, self).__init__()
        channel = in_channel
        convs = []
        for i in range(len(kernels)):
            assert (kernels[i] % 2 == 1)
            convs += [SparseConvBlock(channel, mid_channel, kernels[i], padding=(kernels[i]-1)//2)]
            channel = mid_channel
        self.sparse_convs = nn.Sequential(*convs)
        self.mask_conv = nn.Conv2d(mid_channel+1, out_channel, 1)

    def forward(self, x):
        m = (x > 0).detach().float()
        x, m = self.sparse_convs((x, m))
        x = torch.cat((x, m), dim=1)
        x = self.mask_conv(x)
        # x = F.relu(x, inplace=True)
        return x


class SparseCNN(BaseModel):

    def __init__(self, num_channels=2):
        super().__init__()

        self.sconv1 = SparseConvBlock(1, num_channels, 5, padding=2)
        self.sconv2 = SparseConvBlock(num_channels, num_channels, 5, padding=2)
        self.sconv3 = SparseConvBlock(num_channels, num_channels, 5, padding=2)

        self.smaxpool1 = SparseMaxPooling(2)

        self.sconv21 = SparseConvBlock(num_channels, num_channels, 5, padding=2)
        self.sconv22 = SparseConvBlock(num_channels, num_channels, 5, padding=2)

        self.smaxpool2 = SparseMaxPooling(2)

        self.sconv31 = SparseConvBlock(num_channels, num_channels, 5, padding=2)

        self.smaxpool3 = SparseMaxPooling(2)

        self.sconv41 = SparseConvBlock(num_channels, num_channels, 5, padding=2)

        self.upsample3 = SparseUpsampling(scale_factor=2, align_corners=True)
        self.concat3 = SparseConcatConv(2*num_channels, num_channels)
        self.sconv4 = SparseConvBlock(num_channels, num_channels, 3,  padding=1)

        self.upsample2 = SparseUpsampling(scale_factor=2, align_corners=True)
        self.concat2 = SparseConcatConv(2*num_channels, num_channels)
        self.sconv5 = SparseConvBlock(num_channels, num_channels, 3, padding=1)

        self.upsample1 = SparseUpsampling(scale_factor=2, align_corners=True)
        self.concat1 = SparseConcatConv(2*num_channels, num_channels)
        self.sconv6 = SparseConvBlock(num_channels, num_channels, 3, padding=1)

        self.conv1 = nn.Conv2d(num_channels+1, num_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(num_channels, 1, 1)

    def forward(self, input0):
        down1 = self.sconv1(input0)
        down1 = self.sconv2(down1)
        down1 = self.sconv3(down1)

        # Downsample 1
        down2 = self.smaxpool1(down1)
        down2 = self.sconv21(down2)
        down2 = self.sconv22(down2)

        # Downsample 2
        down3 = self.smaxpool2(down2)
        down3 = self.sconv31(down3)

        # Downsample 3
        down4 = self.smaxpool3(down3)
        down4 = self.sconv41(down4)

        # Upsample 1
        up3 = self.upsample3(down4)
        out3 = self.sconv4(self.concat3(down3, up3))

        # Upsample 2
        up2 = self.upsample2(out3)
        out2 = self.sconv5(self.concat2(down2, up2))

        # Upsample 3
        up1 = self.upsample1(out2)
        out1 = self.sconv6(self.concat1(down1, up1))

        out = self.conv_out(self.conv1(torch.cat(out1, 1)))

        return out, out1[1]


if __name__ == '__main__':
    scnn = SparseCNN(num_channels=8)
    print(scnn.__str__())
    x0 = torch.rand(1, 1, 480, 752)
    c0 = (x0 > 0.5).float()
    xout = scnn((x0, c0))