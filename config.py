from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

# Whether is debug mode
__C.DEBUG = False

# A small number that's used many times
__C.EPS = 1e-14

# Maximal number of gt rois in an image during Training
__C.MAX_NUM_GT_BOXES = 5

# Max window size of RPN
__C.MAX_WINDOW_SIZE = 170

__C.MIN_WINDOW_SIZE = 40 
#=============================================================================#
#
# Training options
#
__C.TRAIN = edict()


# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True
# Only train rpn
__C.TRAIN.RPN_EPOCH = 21

#=============================================================================#
# Use RPN to detect objects
__C.TRAIN.HAS_RPN = True
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.8
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.8
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 100
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_MIN_SIZE = 2
# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = 1.0

#=============================================================================#
# Parameter for Loss function
__C.LOSS = edict()

#=============================================================================#
#
# Testing options
#
__C.TEST = edict()


# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Propose boxes
__C.TEST.HAS_RPN = False

#=============================================================================#

def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = __C
    for subkey in key_list[:-1]:
      assert subkey in d
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d
    try:
      value = literal_eval(v)
    except:
      # handle the case when v is a string literal
      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
        type(value), type(d[subkey]))
    d[subkey] = value
