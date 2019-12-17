from __future__ import absolute_import, division, print_function

import time

import torch
import torch.nn as nn
from progress.bar import Bar

from utils.utils import AverageMeter


class BaseTrainer(object):
    def __init__(
        self, cfg, local_rank, model, optimizer=None):
        self.cfg = cfg
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(cfg, local_rank)
        self.model = model
        self.local_rank = local_rank

    def set_device(self, gpus, chunk_sizes, device):
    
        if  self.cfg.TRAIN.DISTRIBUTE:
            self.model = self.model.to(device)
            self.model = nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True,
                                                        device_ids=[self.local_rank, ],
                                                        output_device=self.local_rank)
        else:
            self.model = nn.DataParallel(self.model).to(device)
        self.loss.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
    
        model = self.model    
        if phase == 'train':
            self.model.train()
        else:
            if len(self.cfg.GPUS) > 1:
                model = model.module        
            model.eval()
            torch.cuda.empty_cache()

        cfg = self.cfg
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        bar = Bar('{}/{}'.format(cfg.TASK, cfg.EXP_ID), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=torch.device('cuda:%d'%self.local_rank), non_blocking=True)    
            
            outputs = model(batch['input'])
            loss, loss_stats = self.loss(outputs, batch)
            
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                  loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not cfg.TRAIN.HIDE_DATA_TIME:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                    '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if cfg.PRINT_FREQ > 0:
                if iter_id % cfg.PRINT_FREQ == 0:
                    print('{}/{}| {}'.format(cfg.TASK, cfg.EXP_ID, Bar.suffix)) 
            else:
                bar.next()
      
            if cfg.DEBUG > 0:
                self.debug(batch, outputs, iter_id)
      
            if phase == 'val':
                self.save_result(outputs, batch, results)
            del outputs, loss, loss_stats
    
        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, cfg):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
