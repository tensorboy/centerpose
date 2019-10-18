from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.multi_pose import MultiPoseDataset
from .dataset.coco_hp import COCOHP


dataset_factory = {
  'coco_hp': COCOHP
}

_sample_factory = {
  'multi_pose': MultiPoseDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
