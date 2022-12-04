import torch.nn as nn
import torch

from ..nn import LazyModule, Module


class Classifier(Module):
    def __init__(self, encoder, num_classes=1000):
        super().__init__()

        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Flatten(), LazyModule(
                nn.Linear, {"out_features": num_classes},
                arg_name="in_features", dim=1
            )
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)

        return x

