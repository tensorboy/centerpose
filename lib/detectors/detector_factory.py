from __future__ import absolute_import, division, print_function

from .multi_pose import MultiPoseDetector

detector_factory = {
  'multi_pose': MultiPoseDetector, 
}
