import torch.nn as nn
import torch

import torch.nn.functional as F
from torch.autograd import Variable

import math

__all__ = ["EfficientNet_bb", "efficientnet_b0_bb", "efficientnet_b1_bb",
           "efficientnet_b2_bb", "efficientnet_b3_bb", "efficientnet_b4_bb",
           "efficientnet_b5_bb", "efficientnet_b6_bb", "efficientnet_b7_bb"]


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


def roundChannels(c, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    
    return new_c

def roundRepeats(r):
    return int(math.ceil(r))

def dropPath(x, drop_probability, training):
    if drop_probability > 0 and training:
        keep_probability = 1 - drop_probability
        if x.is_cuda:
            mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_probability))
        else:
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_probability))

        x.div_(keep_probability)
        x.mul_(mask)

    return x

def batchNorm(channels, eps=1e-3, momentum=0.01):
    return nn.BatchNorm2d(channels, eps=eps, momentum=momentum)

#CONV3x3
def conv3x3(in_channel, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channels, 3, stride, 1, bias=False),
        batchNorm(out_channels),
        Swish()
    )

#CONV1x1
def conv1x1(in_channel, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channels, 1, 1, 0, bias=False),
        batchNorm(out_channels),
        Swish()
    )

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel, squeeze_channel, se_ratio):
        super().__init__()
        squeeze_channel = squeeze_channel * se_ratio
        if not squeeze_channel.is_integer():
            raise ValueError('channels must be divisible by 1/se_ratio')

        squeeze_channel = int(squeeze_channel)
        self.se_reduce = nn.Conv2d(channel, squeeze_channel, 1, 1, 0, bias=True)
        self.non_linear1 = Swish()
        self.se_excite = nn.Conv2d(squeeze_channel, channel, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear1(self.se_excite(y))
        y = x * y
        return y

class MBConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, expand_ratio, se_ratio, drop_path_rate):
        super().__init__()
        expand = (expand_ratio != 1)
        expand_channel = in_channel * expand_ratio
        se = (se_ratio != 0)
        self.residual_connection = (stride == 1 and in_channel == out_channel)
        self.drop_path_rate = drop_path_rate

        conv=[]

        if expand:
            pw_expansion = nn.Sequential(
                nn.Conv2d(in_channel, expand_channel, 1, 1, 0, bias=False),
                batchNorm(expand_channel),
                Swish()
            )
            conv.append(pw_expansion)

        #depthwise convolution
        dw = nn.Sequential(
            nn.Conv2d(expand_channel, expand_channel, kernel_size, stride, kernel_size//2, groups=expand_channel, bias=False),
            batchNorm(expand_channel),
            Swish()
        )
        conv.append(dw)

        if se:
            squeeze_excite = SqueezeAndExcitation(expand_channel, in_channel, se_ratio)
            conv.append(squeeze_excite)
        
        pw_projection = nn.Sequential(
            nn.Conv2d(expand_channel, out_channel, 1, 1, 0, bias=False),
            batchNorm(out_channel)
        )
        conv.append(pw_projection)
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            return x + dropPath(self.conv(x), self.drop_path_rate, self.training)
        else:
            return self.conv(x)


class EfficientNet_bb(nn.Module):
    cfg = [
        #(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats)
        [32,  16,  3, 1, 1, 0.25, 1],
        [16,  24,  3, 2, 6, 0.25, 2],
        [24,  40,  5, 2, 6, 0.25, 2],
        [40,  80,  3, 2, 6, 0.25, 3],
        [80,  112, 5, 1, 6, 0.25, 3],
        [112, 192, 5, 2, 6, 0.25, 4],
        [192, 320, 3, 1, 6, 0.25, 1]
    ]

    def __init__(self, input_channels, param, stem_channels=32, feature_size=1280, drop_connect_rate=0.2):
        super().__init__()

        # scaling width 
        width_coefficient = param[0]
        if width_coefficient != 1.0:
            stem_channels = roundChannels(stem_channels * width_coefficient)
            for conf in self.cfg:
                conf[0] = roundChannels(conf[0] * width_coefficient)
                conf[1] = roundChannels(conf[1] * width_coefficient)

        # scaling depth
        depth_coefficient = param[1]
        if depth_coefficient != 1.0:
            for conf in self.cfg:
                conf[6] = roundRepeats(conf[6] * depth_coefficient)

        #scaling resolution
        input_size = param[2]

        self.stem_conv = conv3x3(input_channels, stem_channels, 2)

        #total blocks
        total_blocks = 0
        for conf in self.cfg:
            total_blocks += conf[6]

        blocks = []
        for in_channel, out_channel, kernel_size, stride, expand_ratio, se_ratio, repeats in self.cfg:
            
            drop_rate = drop_connect_rate * (len(blocks) /  total_blocks)
            blocks.append(MBConvBlock(in_channel, out_channel, kernel_size, stride, expand_ratio, se_ratio, drop_rate))
            for _ in range(repeats-1):
                drop_rate = drop_connect_rate * (len(blocks) / total_blocks)
                blocks.append(MBConvBlock(out_channel, out_channel, kernel_size, 1, expand_ratio, se_ratio, drop_rate))
        self.blocks = nn.Sequential(*blocks)

        self.head_conv = conv1x1(self.cfg[-1][1], feature_size)
        #self.avgpool = nn.AvgPool2d(input_size//32, stride=1)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        x = torch.mean(x, (2, 3))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# param = {
#         # 'efficientnet type': (width_coef, depth_coef, resolution, dropout_rate)
#         'efficientnetb0': (1.0, 1.0, 224, 0.2), 'efficientnetb1': (1.0, 1.1, 240, 0.2),
#         'efficientnetb2': (1.1, 1.2, 260, 0.3), 'efficientnetb3': (1.2, 1.4, 300, 0.3),
#         'efficientnetb4': (1.4, 1.8, 380, 0.4), 'efficientnetb5': (1.6, 2.2, 456, 0.4),
#         'efficientnetb6': (1.8, 2.6, 528, 0.5), 'efficientnetb7': (2.0, 3.1, 600, 0.5)
#     }


def efficientnet_b0_bb(input_channels=3, **kwargs) -> EfficientNet_bb:
    model = EfficientNet_bb(input_channels, (1.0, 1.0, 224, 0.2), **kwargs)
    return model

def efficientnet_b1_bb(input_channels=3, **kwargs) -> EfficientNet_bb:
    model = EfficientNet_bb(input_channels, (1.0, 1.1, 240, 0.2), **kwargs)
    return model

def efficientnet_b2_bb(input_channels=3, **kwargs) -> EfficientNet_bb:
    model = EfficientNet_bb(input_channels, (1.1, 1.2, 260, 0.3), **kwargs)
    return model

def efficientnet_b3_bb(input_channels=3, **kwargs) -> EfficientNet_bb:
    model = EfficientNet_bb(input_channels, (1.2, 1.4, 300, 0.3), **kwargs)
    return model

def efficientnet_b4_bb(input_channels=3, **kwargs) -> EfficientNet_bb:
    model = EfficientNet_bb(input_channels, (1.4, 1.8, 380, 0.4), **kwargs)
    return model

def efficientnet_b5_bb(input_channels=3, **kwargs) -> EfficientNet_bb:
    model = EfficientNet_bb(input_channels, (1.6, 2.2, 456, 0.4), **kwargs)
    return model

def efficientnet_b6_bb(input_channels=3, **kwargs) -> EfficientNet_bb:
    model = EfficientNet_bb(input_channels, (1.8, 2.6, 528, 0.5), **kwargs)
    return model

def efficientnet_b7_bb(input_channels=3, **kwargs) -> EfficientNet_bb:
    model = EfficientNet_bb(input_channels, (2.0, 3.1, 600, 0.5), **kwargs)
    return model
