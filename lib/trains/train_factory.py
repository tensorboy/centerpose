from __future__ import absolute_import, division, print_function

from .multi_pose import MultiPoseTrainer


train_factory = {
  'multi_pose': MultiPoseTrainer, 
}
