import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torchvision.models import resnet

from base import BaseModel
from models.nconv import NormCNN

path_to_dir = os.path.abspath(os.path.dirname(__file__))

class FeatureNet(nn.Module):

    def __init__(self, pretrained=None):
        super().__init__()
        self.d_net = NormCNN(pos_fn='softplus')
        if pretrained is not None:
            checkpoint = torch.load(pretrained)
            self.d_net.load_state_dict(checkpoint['state_dict'])

            # Disable Training for the unguided module
            for p in self.d_net.parameters():
                p.requires_grad = False

        self.d = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
        )  # 11,664 Params

        # RGB stream
        self.rgb = nn.Sequential(
            nn.Conv2d(4, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
        )  # 186,624 Params

        # Fusion stream
        self.fuse = nn.Sequential(
            nn.Conv2d(80, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU()
            )

        self.depth_pred = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 32, 3, 1, 1),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 1, 1, 1))

        self.score_pred = nn.Sequential(ResidualBlock(32),
                                        ResidualBlock(32),
                                        nn.Conv2d(32, 1, 1, 1),
                                        nn.Sigmoid())

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, x0_d, c0, x0_rgb):

        # Depth Network
        xout_d, cout_d = self.d_net(x0_d, c0)
        # xout_d = self.d_net(x0_d)  # , c0)

        # self.xout_d = x_d #xout_d
        # cout_d = c_d # cout_d

        # Extract depth features
        xout_d = self.d(xout_d)

        # RGB network
        xout_rgb = self.rgb(torch.cat((x0_rgb, cout_d), 1))
        # xout_rgb = self.rgb(x0_rgb)
        # self.xout_rgb = xout_rgb

        # Fusion Network
        features = self.fuse(torch.cat((xout_rgb, xout_d), 1))
        depth = self.depth_pred(features)
        score = self.score_pred(features)

        return features, depth, score


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # m.weight.data.normal_(0, 1e-3)
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            # m.bias.data.zero_()
            nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.ConvTranspose2d):
        # m.weight.data.normal_(0, 1e-3)
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            # m.bias.data.zero_()
            nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        # layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.ReLU(inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights:
    for m in layers.modules():
        init_weights(m)

    return layers


def convt_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights:
    for m in layers.modules():
        init_weights(m)

    return layers


class DepthCompletionNet(BaseModel):
    def __init__(self, pretrained_ncnn=None, bn=False):
        super(DepthCompletionNet, self).__init__()

        self.bn = bn
        # self.feature_root = feature_root

        self.stage1 = FeatureNet(pretrained=pretrained_ncnn)

        # Fusion stream
        self.warped_depth_features = nn.Sequential(conv_bn_relu(1, 32, kernel_size=3, padding=1, stride=1, bn=self.bn),
                                                   ResidualBlock(32),
                                                   ResidualBlock(32),
                                                   ResidualBlock(32))

        self.refine = nn.Sequential(UNet(32, 32, 32, 3, batchnorms=self.bn))
        self.pred_depth = nn.Sequential(conv_bn_relu(64, 32, kernel_size=3, padding=1, stride=1, bn=self.bn),
                                        ResidualBlock(32),
                                        ResidualBlock(32),
                                        nn.Conv2d(32, 1, 1, 1))

        self.pred_cfd = nn.Sequential(conv_bn_relu(65, 32, kernel_size=3, padding=1, stride=1, bn=self.bn),
                                      ResidualBlock(32),
                                      ResidualBlock(32),
                                      nn.Conv2d(32, 1, 1, 1),
                                      nn.Sigmoid())

    def forward(self, inputs, prev_state=None, crop_at=None, id_img=None):
        # (img has shape: (batch_size, h, w)) (grayscale)
        # (sparse has shape: (batch_size, h, w))
        imgs, sdmaps = inputs
        cfd0 = (sdmaps > 0)
        cfd0 = cfd0.float()
        warped_depth, warped_cfd = prev_state

        cur_features, cur_depth, cur_score = self.stage1(sdmaps, cfd0, imgs)

        warped_features = self.warped_depth_features(warped_depth)
        # cur_weight, warped_weight = torch.chunk(F.softmax(torch.cat((cur_score, warped_cfd), 1), dim=1), 2, dim=1)
        cur_weight, warped_weight = torch.chunk(F.normalize(torch.cat((cur_score, warped_cfd), 1), p=1, dim=1), 2, dim=1)

        itg_features = warped_weight * warped_features + cur_weight * cur_features
        itg_features = self.refine(itg_features)

        # depth prediction
        input_depth = torch.cat((itg_features, cur_features), 1)
        depth = self.pred_depth(input_depth)

        # confidence prediction
        input_cfd = torch.cat((itg_features, cur_features, depth), 1)
        input_cfd = input_cfd.detach()
        cfd = self.pred_cfd(input_cfd)

        return depth, cfd, cur_depth, cur_score


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, bn=False):
        super(ResidualBlock, self).__init__()
        layers = [nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)]
        if bn:
            layers.append(nn.BatchNorm2d(channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True)) #nn.ReLU())
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
        if bn:
            layers.append(nn.BatchNorm2d(channels, affine=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = out + residual
        return out


class UNet(torch.nn.Module):
    """
    Basic UNet building block, calling itself recursively.
    Note that the final output does not have a ReLU applied.
    """

    def __init__(self, Cin, F, Cout, depth, batchnorms=True):
        super().__init__()
        self.F = F
        self.depth = depth

        if batchnorms:
            self.pre = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(Cin, F, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(F),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(F),
                torch.nn.ReLU(),
            )

            self.post = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(3 * F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(F),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, Cout, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(Cout),
            )
        else:
            self.pre = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(Cin, F, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
            )

            self.post = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(2 * F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, Cout, kernel_size=3, stride=1, padding=0),
            )

        if depth > 1:
            self.process = UNet(F, 2 * F, F, depth - 1, batchnorms=batchnorms)
        else:
            if batchnorms:
                self.process = torch.nn.Sequential(
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(F, 2 * F, kernel_size=3, stride=1, padding=0),
                    torch.nn.BatchNorm2d(2 * F),
                    torch.nn.ReLU(),
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(2 * F, 2 * F, kernel_size=3, stride=1, padding=0),
                    torch.nn.BatchNorm2d(2 * F),
                    torch.nn.ReLU(),
                )
            else:
                self.process = torch.nn.Sequential(
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(F, F, kernel_size=3, stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(F, F, kernel_size=3, stride=1, padding=0),
                    torch.nn.ReLU(),
                )

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, data):
        features = self.pre(data)
        lower_scale = self.maxpool(features)
        lower_features = self.process(lower_scale)
        upsampled = F.interpolate(lower_features, scale_factor=2, mode="bilinear",
                                                    align_corners=False)
        H = data.shape[2]
        W = data.shape[3]
        upsampled = upsampled[:, :, :H, :W]
        result = self.post(torch.cat((features, upsampled), dim=1))

        return result

    def get_influence_percentages(self):
        """
        This function is intended to return a matrix of influences.
        I.e. for each output channel it returns the percentage it is controlled by each input channel.
        Very very roughly speaking, as all this does is iteratively calculate these percentages based on fractional absolute weighting.
        Output:
            percentages -- C_out x C_in matrix giving the weights
        """
        if isinstance(self.pre[1], torch.nn.BatchNorm2d):
            print("BatchNorm UNets not supported for influence percentages")
            return None
        pre1 = self.pre[1].weight.abs().sum(dim=3).sum(dim=2)
        pre1 = pre1 / pre1.sum(dim=1, keepdim=True)
        pre2 = self.pre[4].weight.abs().sum(dim=3).sum(dim=2)
        pre2 = pre2 / pre2.sum(dim=1, keepdim=True)
        pre2 = torch.matmul(pre2, pre1)
        if isinstance(self.process, UNet):
            process2 = torch.matmul(self.process.get_influence_percentages(), pre2)
        else:
            process1 = self.process[1].weight.abs().sum(dim=3).sum(dim=2)
            process1 = process1 / process1.sum(dim=1, keepdim=True)
            process1 = torch.matmul(process1, pre2)
            process2 = self.process[4].weight.abs().sum(dim=3).sum(dim=2)
            process2 = process2 / process2.sum(dim=1, keepdim=True)
            process2 = torch.matmul(process2, process1)

        post1 = self.post[1].weight.abs().sum(dim=3).sum(dim=2)
        post1 = post1 / post1.sum(dim=1, keepdim=True)
        post1 = torch.matmul(post1, torch.cat((pre2, process2), dim=0))
        post2 = self.post[4].weight.abs().sum(dim=3).sum(dim=2)
        post2 = post2 / post2.sum(dim=1, keepdim=True)
        post2 = torch.matmul(post2, post1)

        return post2


if __name__ == '__main__':
    imgs = torch.rand((1, 3, 480, 752)).float().cuda()
    sdmaps = torch.rand((1, 1, 480, 752)).float().cuda()
    cfds_0 = torch.rand((1, 1, 480, 752)).float().cuda()
    Es = torch.eye(4).unsqueeze(0).cuda()
    Ks = torch.rand((1, 3, 3)).float().cuda()
    m = DepthCompletionNet()
    print(m)
    m = m.cuda()
    prev_state = torch.zeros(1, 1, 480, 752).float().cuda(), torch.zeros(1, 1, 480, 752).float().cuda()
    depth, cfd, init_depth, init_cfd = m((imgs, sdmaps), prev_state=prev_state)
    print(depth.size(), cfd.size(), init_depth.size(), init_cfd.size())






