#!/usr/bin/env python

# Copyrigh 2018 houjingyong@gmail.com

# MIT Licence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from anchor_target_layer import AnchorTargetLayer
from config import cfg
from bbox_transform import clip_boxes, get_small_big_boxes 
from loss import smooth_l1_loss

import numpy as np
import random
import sys
import time
import pdb
import time

class RPN_KWS(nn.Module):
    "RPN KWS network"
    def __init__(self, feature_extractor, rpn_nnet, output_dim):
        super(RPN_KWS, self).__init__()
        # basic info of RPN_KWS
        self.training = True
        self.num_class = output_dim

        # three module of RPN_KWS
        self.feature_extractor = feature_extractor
        self.rpn_nnet = rpn_nnet

        # used to get anchor target
        self.anchor_target_layer = AnchorTargetLayer()

        # loss of RPN_KWS
        self.KWS_loss_cls = 0
        self.KWS_loss_bbox = 0

        self.rpn_loss_cls = 0
        self.rpn_loss_bbox = 0

    def forward(self, epoch, speech_data, act_lens, gt_boxes, num_boxes):
        cfg_key = 'TRAIN' if self.training else 'TEST'
        batch_size = speech_data.size(0)
        # Feature extraction
        base_feat = self.feature_extractor(speech_data, act_lens)
        # RPN, get the proposals and anchors and predicted scores 
        anchors_per_utt, proposals, rpn_cls_score, rpn_bbox_pred = self.rpn_nnet(base_feat) 
        # here scores didn't go through the softmax
        # batch_size * num_anchors_per_utt * 2 (box_dim or score_dim)
        rois = clip_boxes(proposals, act_lens, batch_size)

        rpn_label = None
        # here we first calculate the rpn loss and then calculate the kws loss
        if self.training:
            # calculate rpn loss
            rpn_data = self.anchor_target_layer(anchors_per_utt, gt_boxes, act_lens) # rpn trainning targets: labels, bbox_regression targets, bbox_inside_wieght, bbox_outside_weight
            rpn_label = rpn_data[0].long().view(-1)
            rpn_keep = rpn_label.ne(-1).nonzero().view(-1)

            rpn_label = torch.index_select(rpn_label, 0, rpn_keep)
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,self.num_class), 0, rpn_keep)
            
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
            self.rpn_loss_bbox = smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, sigma=10, dim=[0,1])
            
            return rois, rpn_cls_score, rpn_label, self.rpn_loss_cls, self.rpn_loss_bbox 
        return rois, rpn_cls_score, anchors_per_utt
