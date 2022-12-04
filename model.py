import sugar as ss
import torch

bb = ss.models.resnext29_16_64_bb()
model = ss.models.Classification(bb, num_classes=8)

print(model(torch.rand(1, 3, 128, 128)).shape, model.numel())