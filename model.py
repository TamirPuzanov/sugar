import sugar as ss
import torch

bb = ss.models.video_encoders.Linear(ss.resnet18_bb(), 512)
model = ss.models.Classifier(bb, 9)

print(model(torch.rand(1, 8, 3, 224, 224)).shape)
print(model.fc)