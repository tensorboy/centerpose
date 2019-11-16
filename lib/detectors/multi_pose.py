from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
  from external.nms import soft_nms_39
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import multi_pose_decode
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class MultiPoseDetector(BaseDetector):
    def __init__(self, cfg):
        super(MultiPoseDetector, self).__init__(cfg)
        self.flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

    def process(self, images, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()
            outputs = self.model(images)
            hm, wh, hps, reg, hm_hp, hp_offset = outputs
                    
            hm = hm.sigmoid_()
            if self.cfg.LOSS.HM_HP and not self.cfg.LOSS.MSE_LOSS:
                hm_hp = hm_hp.sigmoid_()

            reg = reg if self.cfg.LOSS.REG_OFFSET else None
            hm_hp = hm_hp if self.cfg.LOSS.HM_HP else None
            hp_offset = hp_offset if self.cfg.LOSS.REG_HP_OFFSET else None
            torch.cuda.synchronize()
            forward_time = time.time()

            if self.cfg.TEST.FLIP_TEST:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                hps = (hps[0:1] + 
                  flip_lr_off(hps[1:2], self.flip_idx)) / 2
                hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
                        if hm_hp is not None else None
                reg = reg[0:1] if reg is not None else None
                hp_offset = hp_offset[0:1] if hp_offset is not None else None

            dets = multi_pose_decode(
            hm, wh, hps,
            reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.cfg.TEST.TOPK)

        if return_time:
            return outputs, dets, forward_time
        else:
            return outputs, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets = multi_pose_post_process(
          dets.copy(), [meta['c']], [meta['s']],
          meta['out_height'], meta['out_width'])
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 56)
            dets[0][j][:, :4] /= scale
            dets[0][j][:, 5:] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        results[1] = np.concatenate(
            [detection[1] for detection in detections], axis=0).astype(np.float32)
        if self.cfg.TEST.NMS or len(self.cfg.TEST.TEST_SCALES) > 1:
            soft_nms_39(results[1], Nt=0.5, method=2)
        results[1] = results[1].tolist()
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy().copy()
        dets[:, :, :4] *= self.cfg.MODEL.DOWN_RATIO
        dets[:, :, 5:39] *= self.cfg.MODEL.DOWN_RATIO
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
          img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        if self.LOSS.HM_HP:
            pred = debugger.gen_colormap_hp(
                output['hm_hp'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')
  
    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='multi_pose')
        for bbox in results[1]:
            if bbox[4] > self.cfg.TEST.VIS_THRESH:
                debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose')
                debugger.add_coco_hp(bbox[5:39], img_id='multi_pose')
        debugger.show_all_imgs(pause=self.pause)
