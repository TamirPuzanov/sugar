import torch
import torch.nn as nn

from ...nn import Module


class Linear(Module):
    def __init__(self, bb: Module, fc: int = 512) -> None:
        super().__init__()

        self.bb = bb

        self.fc = nn.LazyLinear(fc)
        self.n = fc
    
    def forward(self, x_3d):
        b, a, _, _, _ = x_3d.shape

        r = []

        for t in range(a):
            x = self.bb(x_3d[:, t, :, :, :])
            x = torch.flatten(x, 1)

            r.append(x)
        
        return torch.stack(x, dim=1)

