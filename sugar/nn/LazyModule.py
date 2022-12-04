import torch.nn as nn
import torch

from .Module import Module


class LazyModule(Module):
    def __init__(self, module, kwargs={}, arg_name="in_features", dim=1):
        super().__init__()
        self.module = module
        self.args = kwargs, arg_name
        self.dim = 1

        self.m = None
    
    def forward(self, x):
        if self.m is None:
            self.args[0][self.args[1]] = x.shape[self.dim]
            self.m = self.module(**self.args[0])
        
        return self.m(x)
