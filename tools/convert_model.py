from __future__ import absolute_import, division, print_function

import argparse
import os

import cv2
import torch

import _init_paths
from config import cfg, update_config
from collections import OrderedDict
from detectors.detector_factory import detector_factory
from models.model import create_model, load_model, save_model

config_file = '../experiments/hrnet_w32_512.yaml'

pretrained_model_path = '/home/tensorboy/data/centerpose/model_zoo-20191216T183914Z-001/model_zoo/hardnet_best.pth'
update_config(cfg, config_file)

model = create_model(cfg.MODEL.NAME, cfg.MODEL.HEAD_CONV, cfg)


pretrained_weight = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)['state_dict']   

head_keys = list(pretrained_weight.keys())[-24:]
backbone_keys = list(pretrained_weight.keys())[:-24]

backbone = OrderedDict()
for k in backbone_keys:
    backbone[k] = pretrained_weight[k]
    
model.backbone_model.load_state_dict(backbone)

head = OrderedDict()
for k in head_keys:
    head[k] = pretrained_weight[k]
    
model.head_model.load_state_dict(head)

save_state_dict = OrderedDict()
save_state_dict['epoch'] = 0
save_state_dict['state_dict'] = model.state_dict()
torch.save(save_state_dict, 'hardnet_best.pth')  

