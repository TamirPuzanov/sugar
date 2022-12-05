import sugar as ss
import torch

bb = ss.models.shuffleresnet_50_v2()
model = ss.models.Classifier(bb, 9)

print(model(torch.rand(1, 3, 224, 224)).shape)
print(model.numel())