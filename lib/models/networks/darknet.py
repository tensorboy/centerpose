import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from collections import OrderedDict


def fill_fc_weights(layers):
  for m in layers.modules():
    if isinstance(m, nn.Conv2d):
      nn.init.normal_(m.weight, std=0.001)
      # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
      # torch.nn.init.xavier_normal_(m.weight.data)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)
        
        
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers, head_conv=256):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        self.layer1 = self._make_layer([32, 64], layers[0])
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        #self.layer4 = self._make_layer([256, 512], layers[3])
        #self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256]
        self.hm = nn.Sequential(
                    nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, 1, kernel_size=1, stride=1, padding=0))

                          
        self.wh = nn.Sequential(
                    nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0))
        self.hps = nn.Sequential(
                    nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, 34, kernel_size=1, stride=1, padding=0))                  
        self.reg = nn.Sequential(
                    nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0))        
        self.hm_hp = nn.Sequential(
                    nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, 17, kernel_size=1, stride=1, padding=0))                                           
        self.hp_offset = nn.Sequential(
                        nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0))    

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.hm[-1].bias.data.fill_(-2.19)     
        self.hm_hp[-1].bias.data.fill_(-2.19)                                    
        fill_fc_weights(self.wh)
        fill_fc_weights(self.hps)
        fill_fc_weights(self.reg)
        fill_fc_weights(self.hp_offset) 
        
    def _make_layer(self, planes, blocks):
        layers = []
        #  downsample
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3,
                                stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        #  blocks
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.interpolate(x, size=(128, 128), 
            mode="bilinear", align_corners=True)

        return [self.hm(x), self.wh(x), self.hps(x), self.reg(x), self.hm_hp(x), self.hp_offset(x)]


def darknet21(cfg,is_train=True, **kwargs):
    model = DarkNet([1, 1, 2, 2, 1])
    if is_train and cfg.BACKBONE.INIT_WEIGHTS:
        if isinstance(cfg.BACKBONE.PRETRAINED, str):
            model.load_state_dict(torch.load(cfg.BACKBONE.PRETRAINED))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(cfg.BACKBONE.PRETRAINED))
    return model

def darknet53(num_layers, head_conv, cfg):
    model = DarkNet([1, 2, 8])
    #if is_train and cfg.BACKBONE.INIT_WEIGHTS:
    #    if isinstance(cfg.BACKBONE.PRETRAINED, str):
    #        model.load_state_dict(torch.load(cfg.BACKBONE.PRETRAINED))
    #    else:
    #        raise Exception("darknet request a pretrained path. got [{}]".format(cfg.BACKBONE.PRETRAINED))
    return model
