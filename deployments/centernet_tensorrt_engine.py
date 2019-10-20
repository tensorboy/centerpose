import logging

import os
import cv2
import sys
import numpy as np
from aifi_models.centernet.lib.centernet_utils.post_process import multi_pose_post_process
from aifi_models.centernet.lib.config import _C as cfg
from aifi_models.centernet.lib.centernet_utils.image import get_affine_transform
from ..tensorrt_engine import TensorRTEngine
from ..utils.gcp_utils import create_trt_context
logger = logging.getLogger(__name__)


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = os.path.join(this_dir, 'lib')
add_path(lib_path)

MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('neck'), keypoints.index('right_shoulder')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('neck'), keypoints.index('left_shoulder')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('neck'), keypoints.index('head')]
    ]
    return kp_lines


def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    keypoints = [
        'head',
        'neck',
        'right_shoulder',
        'right_elbow',
        'right_wrist',
        'left_shoulder',
        'left_elbow',
        'left_wrist'

    ]

    return keypoints


_kp_connections = kp_connections(get_keypoints())


def _gather_feat(feat, ind, mask=None):
    dim = feat.shape[2]
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


def post_process(dets, meta, scale=1):
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = multi_pose_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'])
    for j in range(1, 1 + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 29)
        dets[0][j][:, :4] /= scale
        dets[0][j][:, 5:] /= scale
    return dets[0]


def merge_outputs(detections):
    results = {}
    results[1] = np.concatenate(
        [detection[1] for detection in detections], axis=0).astype(np.float32)

    results[1] = results[1].tolist()

    return results


def _topk_channel(scores, K=40):

    batch, cat, height, width = scores.shape
    scores_reshape = scores.reshape(batch, cat, -1)
    topk_inds = (-scores_reshape).argsort()[:, :, :K]

    topk_scores = -np.sort(-scores_reshape)[:, :, :K]

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).astype(np.int).astype(np.float32)
    topk_xs = (topk_inds % width).astype(np.int).astype(np.float32)

    return topk_scores, topk_inds, topk_ys, topk_xs


def _topk(scores, K=40):
    batch, cat, height, width = scores.shape

    scores_reshape = scores.reshape(batch, cat, -1)
    topk_inds = (-scores_reshape).argsort()[:, :, :K]

    topk_scores = -np.sort(-scores_reshape)[:, :, :K]

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).astype(np.int).astype(np.float32)
    topk_xs = (topk_inds % width).astype(np.int).astype(np.float32)

    top_scores_reshape = topk_scores.reshape(batch, -1)
    topk_ind = (-top_scores_reshape).argsort()[:, :K]

    topk_score = -np.sort(-top_scores_reshape)[:, :K]

    topk_clses = (topk_ind / K).astype(np.int)

    topk_inds = _gather_feat(
        topk_inds.reshape(batch, -1, 1), topk_ind).reshape(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


class CenterNetTensorRTEngine(TensorRTEngine):
    def __init__(self, config_file=None, weight_version=1, precision='fp32', max_batch_size=1):
        super(CenterNetTensorRTEngine, self).__init__(config_file, weight_version, precision, max_batch_size)

    def _load_config(self):
        return cfg

    def _initialize_context(self):
        self.context = create_trt_context(
            model_type='centernet', version=self.weight_version,
            precision=self.precision, max_batch_size=self.max_batch_size)

    def _preprocess(self, imgs):
        self.inputs[0].host, self.meta = self.preprocess(imgs)

    def _postprocess(self, batch_size):
        return self.postprocess(self.outputs, batch_size, self.max_batch_size, self.meta)

    @staticmethod
    def preprocess(image, scale=1, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        mean = np.array(cfg.DATASET.MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(cfg.DATASET.STD, dtype=np.float32).reshape(1, 1, 3)

        if cfg.TEST.FIX_RES:
            inp_height, inp_width = cfg.INPUT_H, cfg.INPUT_W
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

        if cfg.TEST.FLIP_TEST:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        meta = {'c': c, 's': s,
                'out_height': inp_height // cfg.MODEL.DOWN_RATIO,
                'out_width': inp_width // cfg.MODEL.DOWN_RATIO}

        return np.ascontiguousarray(images), meta

    @staticmethod
    def postprocess(outputs, batch_size, max_batch_size, meta):

        output = []
        for i in range(len(outputs)):
            output.append(outputs[i].host.reshape(max_batch_size, -1, 128, 128))

        wh, kps, reg, hp_offset, heat, hm_hp, hmax, hm_hp_max = output

        num_joints = cfg.MODEL.NUM_KEYPOINTS
        batch, cat, height, width = heat.shape

        keep = heat == hmax
        heat = hmax * keep

        keep = hm_hp_max == hm_hp
        hm_hp = hm_hp_max * keep

        scores, inds, clses, ys, xs = _topk(heat, K=cfg.TEST.TOPK)

        kps = _tranpose_and_gather_feat(kps, inds)

        kps = kps.reshape(batch, cfg.TEST.TOPK, num_joints * 2)
        kps[..., ::2] += np.repeat(xs.reshape(batch, cfg.TEST.TOPK, 1), num_joints, axis=2)
        kps[..., 1::2] += np.repeat(ys.reshape(batch, cfg.TEST.TOPK, 1), num_joints, axis=2)

        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.reshape(batch, cfg.TEST.TOPK, 2)
        xs = xs.reshape(batch, cfg.TEST.TOPK, 1) + reg[:, :, 0:1]
        ys = ys.reshape(batch, cfg.TEST.TOPK, 1) + reg[:, :, 1:2]

        wh = _tranpose_and_gather_feat(wh, inds)
        wh = wh.reshape(batch, cfg.TEST.TOPK, 2)
        clses = clses.reshape(batch, cfg.TEST.TOPK, 1).astype(np.float32)
        scores = scores.reshape(batch, cfg.TEST.TOPK, 1)

        bboxes = np.concatenate([xs - wh[..., 0:1] / 2,
                                 ys - wh[..., 1:2] / 2,
                                 xs + wh[..., 0:1] / 2,
                                 ys + wh[..., 1:2] / 2], axis=2)

        thresh = 0.1
        kps = kps.reshape(batch, cfg.TEST.TOPK, num_joints, 2).transpose(0, 2, 1, 3)
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

        dist = (np.sum((reg_kps - hm_kps) ** 2, axis=4) ** 0.5)

        min_dist = np.sort(dist)[..., 0]
        min_ind = np.argsort(dist)[..., 0]

        hm_score = np.expand_dims(np.take_along_axis(hm_score, min_ind, 2), -1)

        min_dist = np.expand_dims(min_dist, axis=-1)

        min_ind = np.repeat(min_ind.reshape(batch, num_joints, cfg.TEST.TOPK, 1, 1), 2, axis=-1)
        hm_kps = np.take_along_axis(hm_kps, min_ind, 3)
        hm_kps = hm_kps.reshape(batch, num_joints, cfg.TEST.TOPK, 2)

        left = np.repeat(bboxes[:, :, 0].reshape(batch, 1, cfg.TEST.TOPK, 1), num_joints, axis=1)
        top = np.repeat(bboxes[:, :, 1].reshape(batch, 1, cfg.TEST.TOPK, 1), num_joints, axis=1)
        right = np.repeat(bboxes[:, :, 2].reshape(batch, 1, cfg.TEST.TOPK, 1), num_joints, axis=1)
        bottom = np.repeat(bboxes[:, :, 3].reshape(batch, 1, cfg.TEST.TOPK, 1), num_joints, axis=1)

        mask = (hm_kps[..., 0:1] < left).astype(np.uint8) + \
               (hm_kps[..., 0:1] > right).astype(np.uint8) + \
               (hm_kps[..., 1:2] < top).astype(np.uint8) + \
               (hm_kps[..., 1:2] > bottom).astype(np.uint8) + \
               (hm_score < thresh).astype(np.uint8) + \
               (min_dist > (np.maximum(bottom - top, right - left) * 0.3)).astype(np.uint8)

        mask = np.repeat((mask > 0).astype(np.float32), 2, axis=-1)

        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.transpose(0, 2, 1, 3).reshape(batch, cfg.TEST.TOPK, num_joints * 2)

        dets = np.concatenate([bboxes, scores, kps, hm_score.squeeze(axis=3).transpose(0, 2, 1)], axis=2)

        dets = post_process(dets, meta, 1)

        results = merge_outputs([dets])

        result = []

        all_joints = []
        all_limbs = []
        all_bboxes = []
        all_keypoints = []
        all_bbox_scores = []
        all_keypoint_scores = []

        count = 0
        for bbox in results[1]:
            if bbox[4] > cfg.TEST.VIS_THRESH:
                one_bbox = bbox[:4]
                one_score = bbox[4]
                eight_keypoints = np.array(bbox[5:21]).reshape(num_joints, 2)
                keypoint_scores = bbox[21:29]

                one_joint = []
                global_indices = []
                for jj in range(num_joints):
                    vis = keypoint_scores[jj]
                    if vis > cfg.TEST.KEYPOINT_THRESH:
                        one_joint.append([eight_keypoints[jj][0], eight_keypoints[jj][1], vis, count, jj])
                        global_indices.append(count)
                        count += 1
                    else:
                        eight_keypoints[jj] = [0., 0.]
                        keypoint_scores[jj] = 0.
                        global_indices.append(-1)
                all_joints += one_joint

                one_connection = []
                for ind_a, ind_b in _kp_connections:
                    g_idx_a = global_indices[ind_a]
                    g_idx_b = global_indices[ind_b]
                    if g_idx_a >= 0 and g_idx_b >= 0:
                        vis_a, vis_b = keypoint_scores[ind_a], keypoint_scores[ind_b]
                        connection_score = (vis_a + vis_b) / 2.
                        one_connection.append([g_idx_a, g_idx_b, connection_score])
                all_limbs += one_connection

                all_bbox_scores.append(one_score)
                all_bboxes.append(one_bbox)
                all_keypoints.append(eight_keypoints)
                all_keypoint_scores.append(keypoint_scores)

        batch_result = {
            'all_joints': all_joints,
            'all_limbs': all_limbs
        }

        people = []
        for ii in range(len(all_bboxes)):
            person = {
                'bbox': all_bboxes[ii],
                'bbox_score': all_bbox_scores[ii],
                'score': all_bbox_scores[ii],
                'keypoints': np.asarray(all_keypoints[ii], dtype=np.float32),
                'keypoints_score': np.asarray(all_keypoint_scores[ii], dtype=np.float32).reshape(num_joints, 1)
            }
            people.append(person)

        batch_result['people'] = people
        result.append(batch_result)

        return result
