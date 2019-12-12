from __future__ import absolute_import, division, print_function

import numpy as np

from .image import transform_preds


def multi_pose_post_process(dets, c, s, h, w):
    # dets: batch x max_dets x 40
    # return list of 39 in image coord
    ret = []
    for i in range(dets.shape[0]):
        bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
        pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
        top_preds = np.concatenate(
            [bbox.reshape(-1, 4), dets[i, :, 4:5], 
            pts.reshape(-1, 34), dets[i, :, 39:56]], axis=1).astype(np.float32).tolist()
        ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
    return ret
    
def whole_body_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x 40
    # return list of 39 in image coord
    ret = []
    ind_list = []    
    for i in range(dets.shape[0]):
        bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
        pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
        top_preds = np.concatenate(
            [bbox.reshape(-1, 4), dets[i, :, 4:5], 
            pts.reshape(-1, 34), dets[i, :, 39:56]], axis=1).astype(np.float32).tolist()
        ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})   
        classes = dets[i, :, -1]
        for j in range(num_classes):
          inds = (classes == j)
          ind_list.append(inds)     
    return ret, ind_list
