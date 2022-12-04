import math

import torch.nn as nn
import torch.nn.init as init

from ...nn import Module


__all__ = ['VGG_bb', 'vgg11_bb', 'vgg11_bn_bb', 'vgg13_bb', 'vgg13_bn_bb', 'vgg16_bb', 'vgg16_bn_bb',
           'vgg19_bb', 'vgg19_bn_bb']


class VGG_bb(Module):
    '''
        VGG Model
    '''

    def __init__(self, features):

        super(VGG_bb, self).__init__()
        self.features = features

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        return x

def make_layers(cfg, input_channel=3, batch_norm=False):

    layers = []
    in_channels = input_channel

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}

def vgg11_bb(input_channel=3):
    """VGG 11-layer model (configuration "A")"""
    return VGG_bb(make_layers(cfg['A'], input_channel))


def vgg11_bn_bb(input_channel=3):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG_bb(make_layers(cfg['A'], input_channel, batch_norm=True))


def vgg13_bb(input_channel=3):
    """VGG 13-layer model (configuration "B")"""
    return VGG_bb(make_layers(cfg['B'], input_channel))


def vgg13_bn_bb(input_channel=3):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG_bb(make_layers(cfg['B'], input_channel, batch_norm=True))


def vgg16_bb(input_channel=3):
    """VGG 16-layer model (configuration "D")"""
    return VGG_bb(make_layers(cfg['D'], input_channel))


def vgg16_bn_bb(input_channel=3):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG_bb(make_layers(cfg['D'], input_channel, batch_norm=True))


def vgg19_bb(input_channel=3):
    """VGG 19-layer model (configuration "E")"""
    return VGG_bb(make_layers(cfg['E'], input_channel))


def vgg19_bn_bb(input_channel=3):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG_bb(make_layers(cfg['E'], input_channel, batch_norm=True))