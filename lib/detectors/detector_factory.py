from __future__ import absolute_import, division, print_function

from .multi_pose import MultiPoseDetector
from .whole_body import WholeBodyDetector

detector_factory = {
  'multi_pose': MultiPoseDetector, 
  'whole_body': WholeBodyDetector
}
