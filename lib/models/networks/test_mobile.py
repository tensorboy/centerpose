from __future__ import absolute_import
import torch
from mobilenet import get_mobile_pose_net

model = get_mobile_pose_net(120, 256)

model.cuda()
model.eval()

dummy_input = torch.randn(1, 3, 512, 512).cuda()

output = model(dummy_input)   
print(output.shape)
