from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from progress.bar import Bar

import _init_paths
from config import cfg, update_config
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from external.nms import soft_nms
from logger import Logger
from utils.utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)                          
    args = parser.parse_args()

    return args
    
    
def test_loader(cfg):
    Dataset = get_dataset(cfg.SAMPLE_METHOD, cfg.TASK)

    for batch_data in Dataset:
        pritn(len(batch_data))

if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args.cfg)
    test_loader(cfg)
