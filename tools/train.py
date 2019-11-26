from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import argparse

import torch
import torch.utils.data
import torch.distributed as dist
from models.model import create_model, load_model, save_model
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory

from config import cfg
from config import update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--local_rank', type=int, default=0)                        
    args = parser.parse_args()

    return args
    
def main(cfg, local_rank):
    torch.manual_seed(cfg.SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    Dataset = get_dataset(cfg.SAMPLE_METHOD, cfg.TASK)


    print('Creating model...')
    model = create_model(cfg.MODEL.NAME, cfg.MODEL.HEAD_CONV, cfg)
    
    num_gpus = torch.cuda.device_count()

    if cfg.TRAIN.DISTRIBUTE:
        device = torch.device('cuda:%d'%local_rank)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=num_gpus, rank=local_rank)
    else:
        device = torch.device('cuda')
             
    logger = Logger(cfg)

        
    if cfg.TRAIN.OPTIMIZER=='adam':
        optimizer = torch.optim.Adam(model.parameters(), cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER== 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=0.9)    
    else:
        NotImplementedError
        
    start_epoch = 0
    if cfg.MODEL.LOAD_MODEL != '':
        model, optimizer, start_epoch = load_model(
          model, cfg.MODEL.LOAD_MODEL, optimizer, cfg.TRAIN.RESUME, cfg.TRAIN.LR, cfg.TRAIN.LR_STEP)

    Trainer = train_factory[cfg.TASK]
    trainer = Trainer(cfg, local_rank, model, optimizer)

    cfg.TRAIN.MASTER_BATCH_SIZE

    if cfg.TRAIN.MASTER_BATCH_SIZE == -1:
        master_batch_size = cfg.TRAIN.BATCH_SIZE // len(cfg.GPUS)
    else:
        master_batch_size = cfg.TRAIN.MASTER_BATCH_SIZE
    rest_batch_size = (cfg.TRAIN.BATCH_SIZE - master_batch_size)
    chunk_sizes = [cfg.TRAIN.MASTER_BATCH_SIZE]
    for i in range(len(cfg.GPUS) - 1):
        slave_chunk_size = rest_batch_size // (len(cfg.GPUS) - 1)
        if i < rest_batch_size % (len(cfg.GPUS) - 1):
            slave_chunk_size += 1
        chunk_sizes.append(slave_chunk_size)
    trainer.set_device(cfg.GPUS, chunk_sizes, device)

    print('Setting up data...')
    val_dataset = Dataset(cfg, 'val')
    val_loader = torch.utils.data.DataLoader(
      val_dataset, 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
    )
    
    train_dataset = Dataset(cfg, 'train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                  num_replicas=num_gpus,
                                                                  rank=local_rank)
    train_loader = torch.utils.data.DataLoader(
      train_dataset, 
      batch_size=cfg.TRAIN.BATCH_SIZE//num_gpus if cfg.TRAIN.DISTRIBUTE else cfg.TRAIN.BATCH_SIZE, 
      shuffle=not cfg.TRAIN.DISTRIBUTE,
      num_workers=cfg.WORKERS,
      pin_memory=True,
      drop_last=True,
      sampler = train_sampler if cfg.TRAIN.DISTRIBUTE else None
    )

    print('Starting training...')
    best = 0.
    for epoch in range(start_epoch + 1, cfg.TRAIN.EPOCHS + 1):
        mark = epoch if cfg.TRAIN.SAVE_ALL_MODEL else 'last'
        train_sampler.set_epoch(epoch)
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if cfg.TRAIN.VAL_INTERVALS > 0 and epoch % cfg.TRAIN.VAL_INTERVALS == 0:
            save_model(os.path.join(cfg.OUTPUT_DIR, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
                mAP = val_dataset.run_eval(preds, cfg.OUTPUT_DIR)
                print('mAP is: ', mAP)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if mAP > best:
                best = mAP
                save_model(os.path.join(cfg.OUTPUT_DIR, 'model_best.pth'), 
                           epoch, model)
        else:
            save_model(os.path.join(cfg.OUTPUT_DIR, 'model_last.pth'), 
                     epoch, model, optimizer)
        logger.write('\n')
        if epoch in cfg.TRAIN.LR_STEP:
            save_model(os.path.join(cfg.OUTPUT_DIR, 'model_{}.pth'.format(epoch)), 
                     epoch, model, optimizer)
            lr = cfg.TRAIN.LR * (0.1 ** (cfg.TRAIN.LR_STEP.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
              param_group['lr'] = lr
    logger.close()

if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args.cfg)
    local_rank = args.local_rank
    main(cfg, local_rank)
