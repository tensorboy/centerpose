from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import torch
USE_TENSORBOARD = True
try:
  import tensorboardX
  print('Using tensorboardX')
except:
  USE_TENSORBOARD = False

class Logger(object):
  def __init__(self, cfg):
    """Create a summary writer logging to log_dir."""
    if not os.path.exists(cfg.OUTPUT_DIR):
      os.makedirs(cfg.OUTPUT_DIR)
   
    time_str = time.strftime('%Y-%m-%d-%H-%M')

    file_name = os.path.join(cfg.OUTPUT_DIR, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
      opt_file.write('==> torch version: {}\n'.format(torch.__version__))
      opt_file.write('==> cudnn version: {}\n'.format(
        torch.backends.cudnn.version()))
      opt_file.write('==> Cmd:\n')
      opt_file.write(str(sys.argv))
      opt_file.write('\n==> Opt:\n')
          
    log_dir = cfg.OUTPUT_DIR + '/logs_{}'.format(time_str)
    if USE_TENSORBOARD:
      self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)
    else:
      try:
        os.makedirs(os.path.dirname(log_dir))
      except:
        pass
      try:
        os.makedirs(log_dir)
      except:
        pass
    self.log = open(log_dir + '/log.txt', 'w')
    try:
      os.system('cp {}/opt.txt {}/'.format(cfg.OUTPUT_DIR, log_dir))
    except:
      pass
    self.start_line = True

  def write(self, txt):
    if self.start_line:
      time_str = time.strftime('%Y-%m-%d-%H-%M')
      self.log.write('{}: {}'.format(time_str, txt))
    else:
      self.log.write(txt)  
    self.start_line = False
    if '\n' in txt:
      self.start_line = True
      self.log.flush()
  
  def close(self):
    self.log.close()
  
  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    if USE_TENSORBOARD:
      self.writer.add_scalar(tag, value, step)
