from hardnet import hardnet
import torch

model = hardnet(19).cuda()
inputs = torch.randn((1,3,512,512)).cuda()

outs = model(inputs)

print(outs.shape)

