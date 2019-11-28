from hardnet import hardnet
from darknet import darknet53
import torch

model = hardnet(19).cuda()
inputs = torch.randn((1,3,512,512)).cuda()

outs = model(inputs)

print(outs.shape)



model = darknet53(0,1,2).cuda()

inputs = torch.randn((1,3,512,512)).cuda()

outs = model(inputs)

print(outs.shape)
