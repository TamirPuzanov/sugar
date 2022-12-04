## 🍭Sugar

A simple package to help developers who do not want to spend time writing standard and routine code. Here you will find: model architectures for computer vision, OCR, NLP and other popular tasks, data loaders of popular dataset formats, implementations of various optimization, loss, activation functions and much more!  

```python 
import sugar as ss
import torch

bb = ss.models.resnext29_16_64_bb()
model = ss.models.Classifier(bb, num_classes=8)

print(model(torch.rand(1, 3, 128, 128)).shape)
print(model.numel())
```
