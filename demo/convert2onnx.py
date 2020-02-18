import argparse
import glob
import json
import logging
import math
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
import onnxruntime as nxrun
import torch

import _init_paths
from config import cfg, update_config
from models.decode import multi_pose_decode
from models.model import create_model
from utils.debugger import Debugger
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process

logger = logging.getLogger(__name__)


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

    inp_height, inp_width = cfg.MODEL.INPUT_H, cfg.MODEL.INPUT_W
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0


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
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 56)
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

    model = create_model('res_50', cfg.MODEL.HEAD_CONV, cfg).cuda()

    weight_path = '/home/tensorboy/data/centerpose/trained_best_model/res_50_best_model.pth'
    state_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)['state_dict']
    #model.load_state_dict(state_dict)

    onnx_file_path = "./model/resnet50.onnx"
    
    #img = cv2.imread('test_image.jpg')
    image = cv2.imread('../images/image1.jpeg')
    images, meta = pre_process(image, cfg, scale=1)

    model.cuda()
    model.eval()
    model.float()
    torch_input = images.cuda()
    print(torch_input.shape)
        
    hm, wh, hps, reg, hm_hp, hp_offset = model(torch_input)
    print('hm',hm.shape)
    print('wh',wh.shape)
    print('hps', hps.shape)
    print('reg',reg.shape)
    print('hm_hp',hm_hp.shape)
    print('hp_offset',hp_offset.shape)
            
    torch.onnx.export(model, torch_input, onnx_file_path, verbose=False)
    sess = nxrun.InferenceSession(onnx_file_path)
    
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    
    print(input_name)
    print(sess.get_outputs()[0].name)
    print(sess.get_outputs()[1].name)
    print(sess.get_outputs()[2].name)
    output_onnx = sess.run(None, {input_name:  images.cpu().data.numpy()})
    hm, wh, hps, reg, hm_hp, hp_offset = output_onnx  
    print('hm',hm.shape)
    print('wh',wh.shape)
    print('hps', hps.shape)
    print('reg',reg.shape)
    print('hm_hp',hm_hp.shape)
    print('hp_offset',hp_offset.shape)
  
if __name__ == '__main__':
    config_name = '../experiments/res_50_512x512.yaml'
    update_config(cfg, config_name)
    main(cfg)
