from __future__ import absolute_import, division, print_function

import json
import os
import time

import numpy as np
import pycocotools.coco as coco
import torch.utils.data as data
from pycocotools.cocoeval import COCOeval


class COCOHP(data.Dataset):
    num_classes = 1
    num_joints = 17
    flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
              [11, 12], [13, 14], [15, 16]]
              
    def __init__(self, cfg, split):
        super(COCOHP, self).__init__()

        self.data_dir = os.path.join(cfg.DATA_DIR, 'coco')
        self.img_dir = os.path.join(self.data_dir, 'images', '{}2017'.format(split))
        if split == 'test':
            self.annot_path = os.path.join(
            self.data_dir, 'annotations', 
            'image_info_test-dev2017.json').format(split)
        else:
            self.annot_path = os.path.join(
            self.data_dir, 'annotations', 
            'person_keypoints_{}2017.json').format(split)
        self.max_objs = 32
        self._valid_ids = [1]
        self.class_name = ['__background__', 'person']        
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.split = split
        self.cfg = cfg

        print('==> initializing coco 2017 {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        images = self.coco.getImgIds()
        catIds = self.coco.getCatIds(self.class_name[-1])
        assert catIds == self._valid_ids
        self.images = self.coco.getImgIds(images,catIds)
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            category_id = 1
            for dets in all_bboxes[image_id]:
                bbox = dets[:4]
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                score = dets[4]
                keypoint_prob = np.array(np.array(dets[39:56])>0.1).astype(np.int32).reshape(17,1)
                keypoints = np.array(dets[5:39], dtype=np.float32).reshape(-1, 2)
                #keypoints[np.array(dets[39:56])<0.0]=np.array([0,0])
                #print(keypoint_prob)
                bbox_out  = list(map(self._to_float, bbox))
                keypoints_pred = np.concatenate([
                keypoints, keypoint_prob], axis=1).reshape(51).tolist()
                keypoints_pred  = list(map(self._to_float, keypoints_pred))

                detection = {
                  "image_id": int(image_id),
                  "category_id": int(category_id),
                  "bbox": bbox_out,
                  "score": float("{:.2f}".format(score)),
                  "keypoints": keypoints_pred
                }
                detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results), 
              open('{}/results.json'.format(save_dir), 'w'))


    def run_eval(self, results, save_dir):
        #self.save_results(results, save_dir)
        #seconds = time.time()
        #local_time = time.ctime(seconds).replace(' ', '_').replace(':','_')
        #coco_dets = self.coco.loadRes('{}/{}_results.json'.format(save_dir, local_time))
        coco_dets = self.coco.loadRes(self.convert_eval_format(results))        
        #coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        #coco_eval.evaluate()
        #coco_eval.accumulate()
   
        coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0]
