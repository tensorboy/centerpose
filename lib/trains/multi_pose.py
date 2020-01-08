from __future__ import absolute_import, division, print_function

import numpy as np
import torch

from models.decode import multi_pose_decode
from models.losses import (FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss)
from models.utils import _sigmoid, flip_lr, flip_lr_off, flip_tensor
from utils.debugger import Debugger
from utils.oracle_utils import gen_oracle_map
from utils.post_process import multi_pose_post_process

from .base_trainer import BaseTrainer


class MultiPoseLoss(torch.nn.Module):
    def __init__(self, cfg, local_rank):
        super(MultiPoseLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_hm_hp = torch.nn.MSELoss() if cfg.LOSS.MSE_LOSS else FocalLoss()
        self.crit_kp = RegWeightedL1Loss() if not cfg.LOSS.DENSE_HP else \
                       torch.nn.L1Loss(reduction='sum')
        self.crit_reg = RegL1Loss() if cfg.LOSS.REG_LOSS == 'l1' else \
                        RegLoss() if cfg.LOSS.REG_LOSS == 'sl1' else None                       
        self.cfg = cfg
        self.local_rank = local_rank

    def forward(self, outputs, batch):
        cfg = self.cfg
        hm_loss, wh_loss, off_loss= 0, 0, 0
        hp_loss, off_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0, 0
        hm, wh, hps, reg, hm_hp, hp_offset = outputs

        for s in range(cfg.MODEL.NUM_STACKS):
            hm = _sigmoid(hm)
            if cfg.LOSS.HM_HP and not cfg.LOSS.MSE_LOSS:
                hm_hp = _sigmoid(hm_hp)
      
            if cfg.TEST.EVAL_ORACLE_HMHP:
                hm_hp = batch['hm_hp']
            if cfg.TEST.EVAL_ORACLE_HM:
                hm = batch['hm']
            if cfg.TEST.EVAL_ORACLE_KPS:
                if cfg.LOSS.DENSE_HP:
                    hps = batch['dense_hps']
                else:
                    hps = torch.from_numpy(gen_oracle_map(
                    batch['hps'].detach().cpu().numpy(), 
                    batch['ind'].detach().cpu().numpy(), 
                    cfg.MODEL.OUTPUT_RES, cfg.MODEL.OUTPUT_RES)).to(torch.device('cuda:%d'%self.local_rank))
            if cfg.TEST.EVAL_ORACLE_HP_OFFSET:
                hp_offset = torch.from_numpy(gen_oracle_map(
                hp_offset.detach().cpu().numpy(), 
                batch['hp_ind'].detach().cpu().numpy(), 
                cfg.MODEL.OUTPUT_RES, cfg.MODEL.OUTPUT_RES)).to(torch.device('cuda:%d'%self.local_rank))


            hm_loss += self.crit(hm, batch['hm']) / cfg.MODEL.NUM_STACKS
            if cfg.LOSS.DENSE_HP:
                mask_weight = batch['dense_hps_mask'].sum() + 1e-4
                hp_loss += (self.crit_kp(hps * batch['dense_hps_mask'], 
                                         batch['dense_hps'] * batch['dense_hps_mask']) / 
                                         mask_weight) / cfg.MODEL.NUM_STACKS
            else:
                hp_loss += self.crit_kp(hps, batch['hps_mask'], 
                                    batch['ind'], batch['hps']) / cfg.MODEL.NUM_STACKS
            if cfg.LOSS.WH_WEIGHT > 0:
                wh_loss += self.crit_reg(wh, batch['reg_mask'],
                                     batch['ind'], batch['wh']) / cfg.MODEL.NUM_STACKS
            if cfg.LOSS.REG_OFFSET and cfg.LOSS.OFF_WEIGHT > 0:
                off_loss += self.crit_reg(reg, batch['reg_mask'],
                                      batch['ind'], batch['reg']) / cfg.MODEL.NUM_STACKS
            if cfg.LOSS.REG_HP_OFFSET and cfg.LOSS.OFF_WEIGHT > 0:
                hp_offset_loss += self.crit_reg(
                hp_offset, batch['hp_mask'],
                batch['hp_ind'], batch['hp_offset']) / cfg.MODEL.NUM_STACKS
            if cfg.LOSS.HM_HP and cfg.LOSS.HM_HP_WEIGHT > 0:
                hm_hp_loss += self.crit_hm_hp(
                hm_hp, batch['hm_hp']) / cfg.MODEL.NUM_STACKS
                              
        loss = cfg.LOSS.HM_WEIGHT * hm_loss + cfg.LOSS.WH_WEIGHT * wh_loss + \
               cfg.LOSS.OFF_WEIGHT * off_loss + cfg.LOSS.HP_WEIGHT * hp_loss + \
               cfg.LOSS.HM_HP_WEIGHT * hm_hp_loss + cfg.LOSS.OFF_WEIGHT * hp_offset_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'hp_loss': hp_loss, 
                      'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats

class MultiPoseTrainer(BaseTrainer):
    def __init__(self, cfg, local_rank, model, optimizer=None):
        super(MultiPoseTrainer, self).__init__(cfg, local_rank, model, optimizer=optimizer)

    def _get_losses(self, cfg, local_rank):
        loss_states = ['loss', 'hm_loss', 'hp_loss', 'hm_hp_loss', 
                       'hp_offset_loss', 'wh_loss', 'off_loss']
        loss = MultiPoseLoss(cfg, local_rank)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        cfg = self.cfg
        reg = output[3] if cfg.LOSS.REG_OFFSET else None
        hm_hp = output[4] if cfg.LOSS.HM_HP else None
        hp_offset = output[5] if cfg.LOSS.REG_HP_OFFSET else None
        dets = multi_pose_decode(
          output[0], output[1], output[2], 
          reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=cfg.TEST.TOPK)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

        dets[:, :, :4] *= cfg.MODEL.INPUT_RES / cfg.MODEL.OUTPUT_RES
        dets[:, :, 5:39] *= cfg.MODEL.INPUT_RES / cfg.MODEL.OUTPUT_RES
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= cfg.MODEL.INPUT_RES / cfg.MODEL.OUTPUT_RES
        dets_gt[:, :, 5:39] *= cfg.MODEL.INPUT_RES / cfg.MODEL.OUTPUT_RES
        for i in range(1):
            debugger = Debugger(
            dataset=cfg.SAMPLE_METHOD, ipynb=(cfg.DEBUG==3), theme=cfg.DEBUG_THEME)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
            img * np.array(cfg.DATASET.STD).reshape(1,1,3).astype(np.float32) + cfg.DATASET.MEAN) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output[0][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')

            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > cfg.MODEL.CENTER_THRESH:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                         dets[i, k, 4], img_id='out_pred')
                    debugger.add_coco_hp(dets[i, k, 5:39], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > cfg.MODEL.CENTER_THRESH:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')
                    debugger.add_coco_hp(dets_gt[i, k, 5:39], img_id='out_gt')

            if cfg.LOSS.HM_HP:
                pred = debugger.gen_colormap_hp(output[4][i].detach().cpu().numpy())
                gt = debugger.gen_colormap_hp(batch['hm_hp'][i].detach().cpu().numpy())
                debugger.add_blend_img(img, pred, 'pred_hmhp')
                debugger.add_blend_img(img, gt, 'gt_hmhp')

            if cfg.DEBUG == 4:
                debugger.save_all_imgs(cfg.LOG_DIR, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output[3] if self.cfg.LOSS.REG_OFFSET else None
        hm_hp = output[4] if self.cfg.LOSS.HM_HP else None
        hp_offset = output[5] if self.cfg.LOSS.REG_HP_OFFSET else None
        dets = multi_pose_decode(
          output[0], output[1], output[2], 
          reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.cfg.TEST.TOPK)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

        dets_out = multi_pose_post_process(
          dets.copy(), batch['meta']['c'].cpu().numpy(),
          batch['meta']['s'].cpu().numpy(),
          output[0].shape[2], output[0].shape[3])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = {1: dets_out[0]}
