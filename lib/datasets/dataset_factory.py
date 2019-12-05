from __future__ import absolute_import, division, print_function

from .coco_hp import COCOHP
from .multi_pose import MultiPoseDataset

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
