import sugar as ss
import torch

bb = ss.models.alstm_efficientnet_b0_bb()
model = ss.models.Classifier(bb, 9)

print(model(torch.rand(1, 3, 2, 224, 224)).shape)
print(model(torch.rand(1, 3, 4, 224, 224)).shape)
print(model.numel())