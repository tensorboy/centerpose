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
    parser.add_argument('--NMS',
                        help='whether to do NMS',
                        type=bool,
                        default=False)       
    parser.add_argument('--TESTMODEL',
                        help='model directory',
                        type=str,
                        default='')                                                
    parser.add_argument('--DEBUG', type=int, default=0,
                         help='level of visualization.'
                              '1: only show the final detection results'
                              '2: show the network output features'
                              '3: use matplot to display' # useful when lunching training with ipython notebook
                              '4: save all visualizations to disk')                             
    args = parser.parse_args()

    return args
    
    
def test(cfg):

    Dataset = dataset_factory[cfg.SAMPLE_METHOD]
    Logger(cfg)
    Detector = detector_factory[cfg.TEST.TASK]

    dataset = Dataset(cfg, 'val')
    detector = Detector(cfg)

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(cfg.EXP_ID), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind in range(num_iters):
        img_id = dataset.images[ind]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(dataset.img_dir, img_info['file_name'])
        #img_path = '/home/tensorboy/data/coco/images/val2017/000000004134.jpg'
        ret = detector.run(img_path)

        results[img_id] = ret['results']

        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                       ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
        bar.next()
    bar.finish()
    dataset.run_eval(results, cfg.OUTPUT_DIR)

if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args.cfg)
    cfg.defrost()
    cfg.DEBUG = args.DEBUG
    cfg.TEST.MODEL_PATH = args.TESTMODEL    
    cfg.TEST.NMS = args.NMS    
    cfg.freeze()
    test(cfg)
