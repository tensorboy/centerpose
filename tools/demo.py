from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import argparse

from config import cfg
from config import update_config
from detectors.detector_factory import detector_factory


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    args = parser.parse_args()

    return args

    
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(cfg):
    Detector = detector_factory[cfg.TEST.TASK]
    detector = Detector(cfg)

    if cfg.TEST.DEMO_FILE == 'webcam' or \
        cfg.TEST.DEMO_FILE[cfg.TEST.DEMO_FILE.rfind('.') + 1:].lower() in video_ext:
        cam = cv2.VideoCapture(0 if cfg.TEST.DEMO_FILE == 'webcam' else cfg.TEST.DEMO_FILE)
        detector.pause = False
        while True:
            _, img = cam.read()
            cv2.imshow('input', img)
            ret = detector.run(img)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            if cv2.waitKey(1) == 27:
                return  # esc to quit
    else:
        if os.path.isdir(cfg.TEST.DEMO_FILE):
            image_names = []
            ls = os.listdir(cfg.TEST.DEMO_FILE)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(cfg.TEST.DEMO_FILE, file_name))
        else:
            image_names = [cfg.TEST.DEMO_FILE]
    
        for (image_name) in image_names:
            print(image_name)
            ret = detector.run(image_name)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args.cfg)
    demo(cfg)
