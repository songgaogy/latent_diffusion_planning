# torch stress test
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 32, 3),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3),
).cuda()

opt = torch.optim.Adam(model.parameters())

for i in range(100):
    x = torch.randn(32, 3, 64, 64).cuda()
    y = model(x).mean()
    opt.zero_grad()
    y.backward()
    opt.step()

print("Torch training OK")