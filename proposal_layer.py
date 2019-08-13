from __future__ import absolute_import

import torch 
import torch.nn as nn

from config import cfg
from generate_anchors import AnchorGenerator
from bbox_transform import bbox_transform_inv, clip_boxes

import pdb
import numpy as np
import math
import yaml



class ProposalLayer(nn.Module):
    """
        Outputs object detection proposals by applying estimated bounding-box 
        transfromations to a set of regular boxes (called "anchors")
    """

    def __init__(self, num_anchors_per_frame, min_box_size, max_box_size):
        super(ProposalLayer, self).__init__()
        self.anchor_generator = AnchorGenerator(num_anchors_per_frame, min_box_size, max_box_size)
        self.num_anchors_per_frame = num_anchors_per_frame

    def forward(self, bbox_deltas):
        batch_size = bbox_deltas.size(0)
        feature_len = bbox_deltas.size(1)/self.num_anchors_per_frame
        # First dimension is batchsize, the second dimension is length of 
        # the number of frames
        anchors_per_utt = self.anchor_generator.get_anchors_per_utt(feature_len)
        # anchors for a batch of utterance
        anchors = anchors_per_utt.view(1, self.num_anchors_per_frame * feature_len, 2).expand(batch_size, self.num_anchors_per_frame * feature_len, 2)
        bbox_deltas.reshape(batch_size, self.num_anchors_per_frame * feature_len, 2) 
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        anchors_per_utt = anchors_per_utt.view(self.num_anchors_per_frame *feature_len, 2)
        return anchors_per_utt, proposals
