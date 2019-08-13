#!/usr/bin/env python

# Copyrigh 2018 houjingyong@gmail.com

# MIT Licence

from __future__ import absolute_import 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from RNNs import GRU
from basic_nodes import LinearBlock
from proposal_layer import ProposalLayer

import numpy as np
import random
import sys

from config import cfg

class RPN(nn.Module):
    """
    region proposal network
    """
    def __init__(self, input_dim, num_anchors_per_frame, output_dim):
        super(RPN,self).__init__()
        self.num_class = output_dim
        self.input_dim = input_dim
        self.num_anchors_per_frame = num_anchors_per_frame
        # this is a global value which indicate the number of anchors used in our experiments
        self.min_window_size = cfg.MIN_WINDOW_SIZE
        self.max_window_size = cfg.MAX_WINDOW_SIZE 
 
        self.num_score_out = self.num_anchors_per_frame * self.num_class # 2(bg/fg) * num anchors)
        self.num_bbox_out = self.num_anchors_per_frame * 2 # 2(coords) * num anchors)
        self.cls_score_RPN = nn.Linear(self.input_dim, self.num_score_out, bias=True)
 
        self.bbox_score_RPN = nn.Linear(self.input_dim, self.num_bbox_out, bias=True) 
        
        self.RPN_proposal_layer = ProposalLayer(self.num_anchors_per_frame, self.min_window_size, self.max_window_size)

    def forward(self, x):
        batch_size  = x.size(0)
        feature_len = x.size(1)
        rpn_cls_score = self.cls_score_RPN(x)
        rpn_cls_score = rpn_cls_score.reshape(batch_size, feature_len * self.num_anchors_per_frame, self.num_class)

        rpn_bbox_pred = self.bbox_score_RPN(x)
        rpn_bbox_pred = rpn_bbox_pred.reshape(batch_size, feature_len * self.num_anchors_per_frame, 2)

        anchors_per_utt, proposals = self.RPN_proposal_layer(rpn_bbox_pred)
        return anchors_per_utt, proposals, rpn_cls_score, rpn_bbox_pred

