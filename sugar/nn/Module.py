import torch.nn as nn
import torch


class Module(nn.Module):
    def set_requires_grad(self, v: bool):
        for param in self.parameters():
            param.requires_grad = v
    
    def freeze(self):
        self.set_requires_grad(False)
    
    def unfreeze(self):
        self.set_requires_grad(True)
    
    def numel(self, only_trainable: bool = False):
        """
        Returns the total number of parameters;
        if `only_trainable` is True, then only
        includes parameters with `requires_grad = True`
        """
        parameters = list(self.parameters())
        if only_trainable:
            parameters = [p for p in parameters if p.requires_grad]
        unique = {p.data_ptr(): p for p in parameters}.values()
        return sum(p.numel() for p in unique)