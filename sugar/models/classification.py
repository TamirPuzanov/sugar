import torch.nn as nn
import torch


class Classification(nn.Module):
    def __init__(self, encoder, num_classes=1000):
        super().__init__()

        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_classes)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)

        return x