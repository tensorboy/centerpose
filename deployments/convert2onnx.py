import os
import cv2
import json
import numpy as np
import argparse
import math
import glob
import time
import logging
import _init_paths
import torch
import onnxruntime as nxrun
from utils.post_process import multi_pose_post_process
from utils.image import get_affine_transform
from models.model import create_model
from models.decode import multi_pose_decode
logger = logging.getLogger(__name__)
from utils.debugger import Debugger
from collections import OrderedDict

from config import cfg
from config import update_config

def gather(a, dim, index):
    expanded_index = [index if dim==i else np.arange(a.shape[i]).reshape([-1 if i==j else 1 for j in range(a.ndim)]) for i in range(a.ndim)]
    return a[expanded_index]
    
def gather_numpy(inputs, dim, index):
    """
    Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :param dim: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = inputs.shape[:dim] + inputs.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and self should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(inputs, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)
    
def vgg_preprocess(image):
    image = image.astype(np.float32) / 255.
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = image.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

    preprocessed_img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
    return preprocessed_img

def pre_process(image, cfg=None, scale=1, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    mean = np.array(cfg.DATASET.MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(cfg.DATASET.STD, dtype=np.float32).reshape(1, 1, 3)
    if cfg.TEST.FIX_RES:
      inp_height, inp_width = cfg.MODEL.INPUT_H, cfg.MODEL.INPUT_W
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      inp_height = (new_height | cfg.MODEL.PAD) + 1
      inp_width = (new_width | cfg.MODEL.PAD) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // cfg.MODEL.DOWN_RATIO, 
            'out_width': inp_width // cfg.MODEL.DOWN_RATIO}
    return images, meta

def post_process(dets, meta, scale=1):
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = multi_pose_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'])
    for j in range(1, 1 + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 39)
        dets[0][j][:, :4] /= scale
        dets[0][j][:, 5:] /= scale
    return dets[0]

def merge_outputs(detections, cfg):
    results = {}
    results[1] = np.concatenate(
        [detection[1] for detection in detections], axis=0).astype(np.float32)
    if cfg.TEST.NMS or len(cfg.TEST.TEST_SCALES) > 1:
      soft_nms_39(results[1], Nt=0.5, method=2)
    results[1] = results[1].tolist()
    return results

def show_results(debugger, image, results, cfg):
    debugger.add_img(image, img_id='multi_pose')
    for bbox in results[1]:
      if bbox[4] > cfg.TEST.VIS_THRESH:
        debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose')
        debugger.add_coco_hp(bbox[5:39], img_id='multi_pose')
    debugger.show_all_imgs(pause=True)      
     
     
def _gather_feat(feat, ind, mask=None):
    dim  = feat.shape[2]
    ind = np.repeat(ind[:, :, np.newaxis], dim, axis=2)
    feat = np.take_along_axis(feat, ind, 1) 
    if mask is not None:
        mask = np.expand_dims(mask, 2).reshape(feat.shape)
        feat = feat[mask]
        feat = feat.reshape(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.transpose(0, 2, 3, 1)
    feat = feat.reshape(feat.shape[0], -1, feat.shape[3])
    feat = _gather_feat(feat, ind)
    return feat
       
       
def _topk_channel(scores, K=40):

    batch, cat, height, width = scores.shape
    scores_reshape = scores.reshape(batch, cat, -1)
    topk_inds = (-scores_reshape).argsort()[:, :, :K]

    topk_scores =  -np.sort(-scores_reshape)[:, :, :K]
    
    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).astype(np.int).astype(np.float32)
    topk_xs   = (topk_inds % width).astype(np.int).astype(np.float32)

    return topk_scores, topk_inds, topk_ys, topk_xs

      
def _topk(scores, K=40):
    batch, cat, height, width = scores.shape
    
    scores_reshape = scores.reshape(batch, cat, -1)
    topk_inds = (-scores_reshape).argsort()[:, :, :K]

    topk_scores =  -np.sort(-scores_reshape)[:, :, :K]

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).astype(np.int).astype(np.float32)
    topk_xs   = (topk_inds % width).astype(np.int).astype(np.float32)

 
    top_scores_reshape = topk_scores.reshape(batch, -1)
    topk_ind = (-top_scores_reshape).argsort()[:, :K]    

    topk_score = -np.sort(-top_scores_reshape)[:, :K]        

    topk_clses = (topk_ind / K).astype(np.int)
     
    topk_inds = _gather_feat(
        topk_inds.reshape(batch, -1, 1), topk_ind).reshape(batch, K)
        
    return  topk_score, topk_inds, topk_clses, topk_ys, topk_xs  
      
           
def main(cfg):

    HEADS = dict(zip(cfg.MODEL.HEADS_NAME, cfg.MODEL.HEADS_NUM))
    model = create_model('restrt_50',  OrderedDict(HEADS), 64).cuda()

    weight_path = '/home/tensorboy/data/centerpose/trained_best_model/res_50.pth'
    state_dict = torch.load(weight_path)['state_dict']
    model.load_state_dict(state_dict)

    onnx_file_path = "ckpt1.onnx"
    
    #img = cv2.imread('test_image.jpg')
    image = cv2.imread('../images/17790319373_bd19b24cfc_k.jpg')
    images, meta = pre_process(image, cfg, scale=1)

    model.cuda()
    model.eval()
    model.float()
    torch_input = images.cuda()

    #output_pytorch = model(torch_input)

    torch.onnx.export(model, torch_input, onnx_file_path, verbose=False)
    sess = nxrun.InferenceSession(onnx_file_path)
    
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    output_onnx = sess.run(None, {input_name:  images.cpu().data.numpy()})
    
    heat, hmax, hm_hp, hm_hp_max, kps, reg, hp_offset, wh = output_onnx
    
    torch.cuda.synchronize()
    tic = time.time()
    num_joints = cfg.MODEL.NUM_KEYPOINTS
    batch, cat, height, width = heat.shape
    
    keep = heat==hmax
    heat = hmax*keep
    
    keep = hm_hp_max==hm_hp
    hm_hp = hm_hp_max*keep

    scores, inds, clses, ys, xs = _topk(heat, K=cfg.TEST.TOPK)
    
    kps = _tranpose_and_gather_feat(kps, inds)    

    kps = kps.reshape(batch, cfg.TEST.TOPK, num_joints*2)
    kps[..., ::2] +=np.repeat(xs.reshape(batch, cfg.TEST.TOPK, 1), num_joints, axis=2)
    kps[..., 1::2] +=np.repeat(ys.reshape(batch, cfg.TEST.TOPK, 1), num_joints, axis=2)

    reg = _tranpose_and_gather_feat(reg, inds)
    reg = reg.reshape(batch, cfg.TEST.TOPK, 2)
    xs = xs.reshape(batch, cfg.TEST.TOPK, 1) + reg[:, :, 0:1]
    ys = ys.reshape(batch, cfg.TEST.TOPK, 1) + reg[:, :, 1:2]

    wh = _tranpose_and_gather_feat(wh, inds)
    wh = wh.reshape(batch, cfg.TEST.TOPK, 2)
    clses  = clses.reshape(batch, cfg.TEST.TOPK, 1).astype(np.float32)
    scores = scores.reshape(batch, cfg.TEST.TOPK, 1)

    bboxes = np.concatenate([xs - wh[..., 0:1] / 2, 
                      ys - wh[..., 1:2] / 2,
                      xs + wh[..., 0:1] / 2, 
                      ys + wh[..., 1:2] / 2], axis=2)    
    
        
    thresh = 0.1
    kps = kps.reshape(batch, cfg.TEST.TOPK, num_joints, 2).transpose(0,2,1,3)
    reg_kps = np.repeat(np.expand_dims(kps, 3), cfg.TEST.TOPK, axis=3)

    hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=cfg.TEST.TOPK)

    hp_offset = _tranpose_and_gather_feat(
      hp_offset, hm_inds.reshape(batch, -1))
    hp_offset = hp_offset.reshape(batch, num_joints, cfg.TEST.TOPK, 2)
    hm_xs = hm_xs + hp_offset[:, :, :, 0]
    hm_ys = hm_ys + hp_offset[:, :, :, 1]          

    mask = (hm_score > thresh).astype(np.float32)

    hm_score = (1 - mask) * -1 + mask * hm_score
    hm_ys = (1 - mask) * (-10000) + mask * hm_ys
    hm_xs = (1 - mask) * (-10000) + mask * hm_xs   
    
    
    hm_kps = np.expand_dims(np.stack([hm_xs, hm_ys], axis=-1), axis=2)
    hm_kps = np.repeat(hm_kps, cfg.TEST.TOPK, axis=2)

    dist = (np.sum((reg_kps - hm_kps) ** 2,axis=4) ** 0.5)

    min_dist = np.sort(dist)[...,0]
    min_ind = np.argsort(dist)[...,0]

    hm_score = np.expand_dims(np.take_along_axis(hm_score, min_ind, 2), -1)

    min_dist = np.expand_dims(min_dist, axis= -1)

    min_ind = np.repeat(min_ind.reshape(batch, num_joints, cfg.TEST.TOPK, 1, 1), 2, axis=-1)
    hm_kps = np.take_along_axis(hm_kps, min_ind, 3)
    hm_kps = hm_kps.reshape(batch, num_joints, cfg.TEST.TOPK, 2)

    l = np.repeat(bboxes[:, :, 0].reshape(batch, 1, cfg.TEST.TOPK, 1), num_joints, axis=1)
    t = np.repeat(bboxes[:, :, 1].reshape(batch, 1, cfg.TEST.TOPK, 1), num_joints, axis=1)
    r = np.repeat(bboxes[:, :, 2].reshape(batch, 1, cfg.TEST.TOPK, 1), num_joints, axis=1)
    b = np.repeat(bboxes[:, :, 3].reshape(batch, 1, cfg.TEST.TOPK, 1), num_joints, axis=1)


    mask = (hm_kps[..., 0:1] < l).astype(np.uint8) + (hm_kps[..., 0:1] > r).astype(np.uint8) + \
         (hm_kps[..., 1:2] < t).astype(np.uint8) + (hm_kps[..., 1:2] > b).astype(np.uint8) + \
         (hm_score < thresh).astype(np.uint8) + (min_dist > (np.maximum(b - t, r - l) * 0.3)).astype(np.uint8)
   
    mask = np.repeat((mask > 0).astype(np.float32), 2, axis=-1)
  
    kps = (1 - mask) * hm_kps + mask * kps
    kps = kps.transpose(0, 2, 1, 3).reshape(batch, cfg.TEST.TOPK, num_joints * 2)

    dets = np.concatenate([bboxes, scores, kps, clses], axis=2)

    dets = post_process(dets, meta, 1)

    detections = [dets]
    #dets = torch.from_numpy(output_onnx)

    #dets = post_process(dets, meta, 1)

    #results = merge_outputs([dets], opt)
    
    debugger = Debugger((cfg.DEBUG==3), theme=cfg.DEBUG_THEME, 
               num_classes=cfg.MODEL.NUM_CLASSES, dataset=cfg.SAMPLE_METHOD, down_ratio=cfg.MODEL.DOWN_RATIO)
                   
    results = merge_outputs(detections, cfg)
    
    show_results(debugger, image, results, cfg)
           
    #dets = torch.from_numpy(output_onnx)

    #dets = post_process(dets, meta, 1)

    #results = merge_outputs([dets], opt)
    
    #debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug==3),
    #                    theme=opt.debugger_theme)

    #detections = [dets]
    
    #results = merge_outputs(detections, opt)
    
    #show_results(debugger, image, results, opt)

  
if __name__ == '__main__':
    config_name = '../experiments/res_50_512x512_sgd.yaml'
    update_config(cfg, config_name)
    main(cfg)

