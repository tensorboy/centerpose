from __future__ import absolute_import, division, print_function

from .multi_pose import MultiPoseTrainer
from .whole_body import WholeBodyTrainer


train_factory = {
  'multi_pose': MultiPoseTrainer, 
}
