import torch.nn as nn
import torch

from typing import cast, Dict, List, Union
from torch import Tensor


__all__ = [
    "VGG_bb", "_make_layers",
    "vgg11_bb", "vgg13_bb", "vgg16_bb", "vgg19_bb",
    "vgg11_bn_bb", "vgg13_bn_bb", "vgg16_bn_bb", "vgg19_bn_bb",
]


vgg_cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_layers(vgg_cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: nn.Sequential[nn.Module] = nn.Sequential()
    in_channels = 3
    for v in vgg_cfg:
        if v == "M":
            layers.append(nn.MaxPool2d((2, 2), (2, 2)))
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, (3, 3), (1, 1), (1, 1))
            if batch_norm:
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(True))
            else:
                layers.append(conv2d)
                layers.append(nn.ReLU(True))
            in_channels = v

    return layers


class VGG_bb(nn.Module):
    def __init__(self, vgg_cfg: List[Union[str, int]], batch_norm: bool = False) -> None:
        super(VGG_bb, self).__init__()
        self.features = _make_layers(vgg_cfg, batch_norm)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


def vgg11_bb(**kwargs) -> VGG_bb:
    model = VGG_bb(vgg_cfgs["vgg11"], False, **kwargs)

    return model


def vgg13_bb(**kwargs) -> VGG_bb:
    model = VGG_bb(vgg_cfgs["vgg13"], False, **kwargs)

    return model


def vgg16_bb(**kwargs) -> VGG_bb:
    model = VGG_bb(vgg_cfgs["vgg16"], False, **kwargs)

    return model


def vgg19_bb(**kwargs) -> VGG_bb:
    model = VGG_bb(vgg_cfgs["vgg19"], False, **kwargs)

    return model


def vgg11_bn_bb(**kwargs) -> VGG_bb:
    model = VGG_bb(vgg_cfgs["vgg11"], True, **kwargs)

    return model


def vgg13_bn_bb(**kwargs) -> VGG_bb:
    model = VGG_bb(vgg_cfgs["vgg13"], True, **kwargs)

    return model


def vgg16_bn_bb(**kwargs) -> VGG_bb:
    model = VGG_bb(vgg_cfgs["vgg16"], True, **kwargs)

    return model


def vgg19_bn_bb(**kwargs) -> VGG_bb:
    model = VGG_bb(vgg_cfgs["vgg19"], True, **kwargs)

    return model