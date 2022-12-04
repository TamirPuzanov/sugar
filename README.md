## Exemple

``` import sugar as ss
import torch

bb = ss.models.xception_bb(input_channel=3)
model = ss.models.Classification(bb, num_classes=8)

print(model(torch.rand(1, 3, 128, 128)).shape)
```