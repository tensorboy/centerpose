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
from datasets.dataset_factory import get_dataset
from utils.post_process import multi_pose_post_process, whole_body_post_process
from models.decode import multi_pose_decode, whole_body_decode, _nms, _topk, _transpose_and_gather_feat, _topk_channel
from utils.debugger import Debugger
import ipdb


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)                          
    args = parser.parse_args()

    return args
    
    
SAVE_DIR = '/home/tensorboy/data/coco/images/data_loader_vis'
MEAN = np.array([0.408, 0.447, 0.470]).astype(np.float32)
STD = np.array([0.289, 0.274, 0.278]).astype(np.float32)
    
def test_loader(cfg):

    debugger = Debugger((cfg.DEBUG==3), theme=cfg.DEBUG_THEME, 
               num_classes=cfg.MODEL.NUM_CLASSES, dataset=cfg.SAMPLE_METHOD, down_ratio=cfg.MODEL.DOWN_RATIO)
               
    Dataset = get_dataset(cfg.SAMPLE_METHOD, cfg.TASK)
    val_dataset = Dataset(cfg, 'val')
    val_loader = torch.utils.data.DataLoader(
      val_dataset, 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
    )
    for i, batch_data in enumerate(val_loader):
        input_image = batch_data['input']
        heat = batch_data['hm']
        reg = batch_data['reg']        
        reg_mask = batch_data['reg_mask']
        ind = batch_data['ind']
        wh = batch_data['wh']
        kps = batch_data['hps']
        hps_mask = batch_data['hps_mask']
        seg_feat = batch_data['seg']
        hm_hp = batch_data['hm_hp']
        hp_offset = batch_data['hp_offset']
        hp_ind = batch_data['hp_ind']
        hp_mask = batch_data['hp_mask']
        meta = batch_data['meta']

        for k,v in batch_data.items():
            if type(v)==type(dict()):
                for k1, v1 in v.items():
                    print(k1)
                    print(v1)
            else:
                print(k)
                print(v.shape)
        print(input_image.shape)
        print(hm_hp.shape)
        #handle image
        input_image = input_image[0].numpy().transpose(1,2,0)
        input_image = (input_image*STD)+MEAN
        input_image = input_image*255
        input_image = input_image.astype(np.uint8)
        
        heat = heat.sigmoid_()
        hm_hp = hm_hp.sigmoid_()
        
        num_joints = 17

        K = cfg.TEST.TOPK
        # perform nms on heatmaps
        batch, cat, height, width = heat.size()
        heat = _nms(heat)
        scores, inds, clses, ys, xs = _topk(heat, K=K)

        kps = kps.view(batch, K, num_joints * 2)
        kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
        kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)

        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

        wh = wh.view(batch, K, 2)
        
        #weight = _transpose_and_gather_feat(seg, inds)
        ## you can write  (if weight.size(1)!=seg_feat.size(1): 3x3conv  else 1x1conv ) here to select seg conv.
        ## for 3x3
        #weight = weight.view([weight.size(1), -1, 3, 3])
        pred_seg = seg_feat
            
        clses  = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)

        bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                          ys - wh[..., 1:2] / 2,
                          xs + wh[..., 0:1] / 2, 
                          ys + wh[..., 1:2] / 2], dim=2)
        if hm_hp is not None:
            hm_hp = _nms(hm_hp)
            thresh = 0.1
            kps = kps.view(batch, K, num_joints, 2).permute(
              0, 2, 1, 3).contiguous() # b x J x K x 2
            reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
            hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # b x J x K

            hp_offset = hp_offset.view(batch, num_joints, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]

            mask = (hm_score > thresh).float()
            hm_score = (1 - mask) * -1 + mask * hm_score
            hm_ys = (1 - mask) * (-10000) + mask * hm_ys
            hm_xs = (1 - mask) * (-10000) + mask * hm_xs
            hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
              2).expand(batch, num_joints, K, K, 2)
            dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
            min_dist, min_ind = dist.min(dim=3) # b x J x K
            hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
            min_dist = min_dist.unsqueeze(-1)
            min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
              batch, num_joints, K, 1, 2)
            hm_kps = hm_kps.gather(3, min_ind)
            hm_kps = hm_kps.view(batch, num_joints, K, 2)
            l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
                 (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
                 (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
            mask = (mask > 0).float().expand(batch, num_joints, K, 2)
            kps = (1 - mask) * hm_kps + mask * kps
            kps = kps.permute(0, 2, 1, 3).contiguous().view(
              batch, K, num_joints * 2)
        dets = torch.cat([bboxes, scores, kps, torch.transpose(hm_score.squeeze(dim=3), 1, 2)], dim=2)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

        dets, inds = whole_body_post_process(
          dets.copy(), [meta['c'].numpy()], [meta['s'].numpy()],
          128, 128, 1)
        for j in range(1, cfg.MODEL.NUM_CLASSES + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 56)
            dets[0][j][:, :4] /= 1.
            dets[0][j][:, 5:39] /= 1.
            
        print(pred_seg.shape)
        seg = pred_seg[0]    
        trans = get_affine_transform(meta['c'], meta['s'], 0, ( meta['out_width'], meta['out_height']), inv=1)
        debugger.add_img(image, img_id='multi_pose')
        for j in range(1, self.num_classes + 1):
            for b_id, detection in enumerate(results[j]):        
                bbox = detection[:4]
                bbox_prob = detection[4]
                keypoints = detection[5:39]
                keypoints_prob = detection[39:]
                if bbox_prob > self.cfg.TEST.VIS_THRESH:
                    debugger.add_coco_bbox(bbox, 0, bbox_prob, img_id='multi_pose')
                    segment = seg[b_id].detach().cpu().numpy()

                    segment = cv2.warpAffine(segment, trans,(image.shape[1],image.shape[0]),
                                                 flags=cv2.INTER_CUBIC)
                    w,h = bbox[2:4] - bbox[:2]
                    ct = np.array(
                        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

                    segment_mask = np.zeros_like(segment)
                    pad_rate = 0.3
                    x, y = np.clip([ct[0] - (1 + pad_rate) * w / 2, ct[0] + (1 + pad_rate) * w / 2], 0,
                                   segment.shape[1] - 1).astype(np.int), \
                           np.clip([ct[1] - (1 + pad_rate) * h / 2, ct[1] + (1 + pad_rate) * h / 2], 0,
                                   segment.shape[0] - 1).astype(np.int)
                    segment_mask[y[0]:y[1], x[0]:x[1]] = 1
                    segment = segment_mask*segment
                    debugger.add_coco_seg(segment, img_id='multi_pose')                    
                    debugger.add_coco_hp(keypoints, keypoints_prob, img_id='multi_pose')  
                    
        debugger.show_all_imgs(pause=self.pause)            
            
        save_path = os.path.join(SAVE_DIR, '{}.png'.format(i))
        cv2.imwrite(save_path, input_image)

if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args.cfg)
    test_loader(cfg)
