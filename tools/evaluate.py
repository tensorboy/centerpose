from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from external.nms import soft_nms
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory

from config import cfg
from config import update_config

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
    config_name = '../experiments/dla_34_512x512_adam.yaml'
    update_config(cfg, config_name)
    test(cfg)
