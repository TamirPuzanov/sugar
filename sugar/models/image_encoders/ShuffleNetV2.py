import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn

from ...nn import Module


__all__ = ["ShuffleNetV2", "ShuffleResNetV2", "shufflenet_v2_0_5", "shufflenet_v2_1_0", "shufflenet_v2_1_5", 
           "shufflenet_v2_2_0", "shuffleresnet_50_v2", "shuffleresnet_164_v2"]


class ShuffleUnit(Module):
    def __init__(self, groups):
        super(ShuffleUnit, self).__init__()
        self.groups = groups

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.reshape(n, self.groups, c // self.groups, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(n, c, h, w)
        return x


class ConvBnRelu(Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1):
        super(ConvBnRelu, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups,
                      False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True))

    def forward(self, x):
        return self.conv_bn_relu(x)


class ConvBn(Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1):
        super(ConvBn, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups,
                      False),
            nn.BatchNorm2d(out_channel))

    def forward(self, x):
        return self.conv_bn(x)


class SELayer(Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self.avg_pool(x).view(n, c)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y


class ShuffleNetV2Block(Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation=1, stride=1, shuffle_group=2):
        super(ShuffleNetV2Block, self).__init__()

        pad = (kernel_size // 2) * dilation
        self._stride = stride
        if stride == 1:
            # Split and concat unit
            if in_channel != out_channel:
                raise ValueError('in_c must equal out_c if stride is 1, which is {} and {}.'
                                 .format(in_channel, out_channel))
            branch_channel = (in_channel // 2) + (in_channel % 2)
            self._branch_channel = branch_channel
            self.branch = nn.Sequential(
                ConvBnRelu(branch_channel, branch_channel, 1),
                ConvBn(branch_channel, branch_channel, kernel_size, padding=pad, dilation=dilation,
                       groups=branch_channel),
                ConvBnRelu(branch_channel, branch_channel, 1)
            )
        else:
            # No split and downsample unit
            self.branch_0 = nn.Sequential(
                ConvBnRelu(in_channel, out_channel, 1),
                ConvBn(out_channel, out_channel, kernel_size, stride, padding=pad,
                       dilation=dilation, groups=out_channel),
                ConvBnRelu(out_channel, out_channel, 1)
            )
            self.branch_1 = nn.Sequential(
                ConvBn(in_channel, in_channel, kernel_size, stride, padding=pad, dilation=dilation,
                       groups=in_channel),
                ConvBnRelu(in_channel, out_channel, 1)
            )
        self.shuffle = ShuffleUnit(shuffle_group)

    def forward(self, x):
        if self._stride == 1:
            x_0, x_1 = torch.split(x, self._branch_channel, dim=1)
            out = torch.cat([self.branch(x_0), x_1], dim=1)
        else:
            out = torch.cat([self.branch_0(x), self.branch_1(x)], dim=1)
        out = self.shuffle(out)
        return out


class ShuffleNetV2ResBlock(Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation=1, stride=1,
                 shuffle_group=2, use_se_block=True, se_reduction=16):
        super(ShuffleNetV2ResBlock, self).__init__()

        pad = (kernel_size // 2) * dilation
        self._stride = stride
        self._in_channel = in_channel
        self._out_channel = out_channel
        if stride == 1 and in_channel == out_channel:
            # Split and concat unit
            branch_channel = (in_channel // 2) + (in_channel % 2)
            self._branch_channel = branch_channel
            self._blocks = [
                ConvBnRelu(branch_channel, branch_channel, 1),
                ConvBn(branch_channel, branch_channel, kernel_size, padding=pad, dilation=dilation,
                       groups=branch_channel),
                ConvBnRelu(branch_channel, branch_channel, 1)
            ]
            if use_se_block:
                self._blocks.append(SELayer(branch_channel, se_reduction))
            self.branch = nn.Sequential(*self._blocks)
        else:
            # No split and downsample unit
            self._blocks = [
                ConvBnRelu(in_channel, out_channel, 1),
                ConvBn(out_channel, out_channel, kernel_size, stride, padding=pad,
                       dilation=dilation, groups=out_channel),
                ConvBnRelu(out_channel, out_channel, 1)
            ]
            if use_se_block:
                self._blocks.append(SELayer(out_channel, se_reduction))
            self.branch_0 = nn.Sequential(*self._blocks)
            self.branch_1 = nn.Sequential(
                ConvBn(in_channel, in_channel, kernel_size, stride, padding=pad, dilation=dilation,
                       groups=in_channel),
                ConvBnRelu(in_channel, out_channel, 1)
            )
        self.shuffle = ShuffleUnit(shuffle_group)

    def forward(self, x):
        if self._stride == 1 and self._in_channel == self._out_channel:
            x_0, x_1 = torch.split(x, self._branch_channel, dim=1)
            x_0 = x_0 + self.branch(x_0)
            out = torch.cat([x_0, x_1], dim=1)
        else:
            out = torch.cat([self.branch_0(x), self.branch_1(x)], dim=1)
        out = self.shuffle(out)
        return out


class ShuffleNetV2(Module):
    """
    Class for building ShuffleNetV2 model with [0.5, 1.0, 1.5, 2.0] sizes
    """
    def __init__(self, in_channel, model_scale=1.0,
                 shuffle_group=2):
        super(ShuffleNetV2, self).__init__()

        self.block_def = self._select_channel_size(model_scale)
        cur_channel = 24
        down_size = 4

        # First conv down size
        self.blocks = [('Init_Block',
                        nn.Sequential(
                            ConvBnRelu(in_channel, cur_channel, 3,
                                       stride=2, padding=1),
                            nn.MaxPool2d(3, stride=2, padding=1)
                        ))]

        # Middle shuffle blocks
        for idx, block in enumerate(self.block_def[:-1]):
            out_channel, repeat = block
            self.blocks += [('Stage{}_Block1'.format(idx + 2),
                             ShuffleNetV2Block(cur_channel, out_channel // 2,
                                               3, stride=2,
                                               shuffle_group=shuffle_group))]
            down_size *= 2
            for i in range(repeat - 1):
                self.blocks += [('Stage{}_Block{}'.format(idx + 2, i + 2),
                                 ShuffleNetV2Block(out_channel, out_channel,
                                                   3, shuffle_group=shuffle_group))]
            cur_channel = out_channel
        last_channel = self.block_def[-1][0]
        self.blocks += [('Conv', ConvBnRelu(cur_channel, last_channel, 1))]

        # Avg pool and predict

        self.blocks += [('AvgPool',
                         nn.Sequential(nn.AdaptiveAvgPool2d((1, 1))))]
        
        self.model = nn.Sequential(OrderedDict(self.blocks))

    def _select_channel_size(self, model_scale):
        # [(out_channel, repeat_times), (out_channel, repeat_times), ...]
        if model_scale == 0.5:
            return [(48, 4), (96, 8), (192, 4), (1024, 1)]
        elif model_scale == 1.0:
            return [(116, 4), (232, 8), (464, 4), (1024, 1)]
        elif model_scale == 1.5:
            return [(176, 4), (352, 8), (704, 4), (1024, 1)]
        elif model_scale == 2.0:
            return [(244, 4), (488, 8), (976, 4), (2048, 1)]
        else:
            raise ValueError('Unsupported model size.')

    def forward(self, x):
        out = self.model(x)
        return out


class ShuffleResNetV2(Module):
    """
    Class for building ShuffleNetV2-50 and SE-ShuffleNetV2-164
    """
    def __init__(self, in_channel, model_arch=50,
                 shuffle_group=2, use_se_block=True, se_reduction=16):
        super(ShuffleResNetV2, self).__init__()

        self.block_def = self._select_model_size(model_arch)
        down_size = 2
        self.blocks = []

        # First conv down size
        self.init_block, cur_channel = self._get_init_block(model_arch, in_channel)
        self.blocks += self.init_block

        # Middle shuffle blocks
        for idx, block in enumerate(self.block_def[:-1]):
            out_channel, repeat = block
            down_size *= 2

            if idx == 0:
                self.blocks += [('Stage{}_Block1'.format(idx + 2),
                                 nn.MaxPool2d(3, stride=2, padding=1)),
                                ('Stage{}_Block2'.format(idx + 2),
                                 ShuffleNetV2ResBlock(cur_channel, out_channel // 2,
                                                      3, shuffle_group=shuffle_group,
                                                      use_se_block=use_se_block,
                                                      se_reduction=se_reduction)
                                 )]
                for i in range(repeat - 2):
                    self.blocks += [('Stage{}_Block{}'.format(idx + 2, i + 3),
                                     ShuffleNetV2ResBlock(out_channel, out_channel,
                                                          3, shuffle_group=shuffle_group,
                                                          use_se_block=use_se_block,
                                                          se_reduction=se_reduction
                                                          ))]
            else:
                self.blocks += [('Stage{}_Block1'.format(idx + 2),
                                 ShuffleNetV2ResBlock(cur_channel, out_channel // 2,
                                                      3, stride=2,
                                                      shuffle_group=shuffle_group,
                                                      use_se_block=use_se_block,
                                                      se_reduction=se_reduction
                                                      ))]
                for i in range(repeat - 1):
                    self.blocks += [('Stage{}_Block{}'.format(idx + 2, i + 2),
                                     ShuffleNetV2ResBlock(out_channel, out_channel,
                                                          3, shuffle_group=shuffle_group,
                                                          use_se_block=use_se_block,
                                                          se_reduction=se_reduction
                                                          ))]
            cur_channel = out_channel
        last_channel = self.block_def[-1][0]
        self.blocks += [('Conv', ConvBnRelu(cur_channel, last_channel, 1))]

        # Avg pool
        self.blocks += [('AvgPool',
                         nn.Sequential(nn.AdaptiveAvgPool2d((1, 1))))]

        self.model = nn.Sequential(OrderedDict(self.blocks))

    def _get_init_block(self, model_arch, in_channel):
        out_channel = 64
        if model_arch == 50:
            blocks = [('Init_Block',
                       ConvBnRelu(in_channel, out_channel, 3,
                                  stride=2, padding=1)
                       )]
        elif model_arch == 164:
            blocks = [('Init_Block',
                       nn.Sequential(
                           ConvBnRelu(in_channel, out_channel, 3,
                                      stride=2, padding=1),
                           ConvBnRelu(out_channel, out_channel, 3,
                                      stride=1, padding=1),
                           ConvBnRelu(out_channel, 2 * out_channel, 3,
                                      stride=1, padding=1)
                       ))]
            out_channel *= 2
        else:
            raise ValueError('Support arch [50, 164]')
        return blocks, out_channel

    def _select_model_size(self, model_arch):
        # [(out_channel, repeat_times), (out_channel, repeat_times), ...]
        if model_arch == 50:
            return [(244, 4), (488, 4), (976, 6), (1952, 3), (2048, 1)]
        elif model_arch == 164:
            return [(340, 10), (680, 10), (1360, 23), (2720, 10), (2048, 1)]
        else:
            raise ValueError('Support arch [50, 164]')

    def forward(self, x):
        out = self.model(x)
        return out


def shuffleresnet_50_v2(in_channel=3):
    return ShuffleResNetV2(in_channel=in_channel, model_arch=50)

def shuffleresnet_164_v2(in_channel=3):
    return ShuffleResNetV2(in_channel=in_channel, model_arch=164)


def shufflenet_v2_0_5(in_channel=3):
    return ShuffleNetV2(in_channel=in_channel, model_scale=0.5)

def shufflenet_v2_1_0(in_channel=3):
    return ShuffleNetV2(in_channel=in_channel, model_scale=1.0)

def shufflenet_v2_1_5(in_channel=3):
    return ShuffleNetV2(in_channel=in_channel, model_scale=1.5)

def shufflenet_v2_2_0(in_channel=3):
    return ShuffleNetV2(in_channel=in_channel, model_scale=2.0)
