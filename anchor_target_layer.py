#!/usr/bin/env python
#
# Copyright 2018 houjingyong@gmail.com
#
# Lisence MIT
#

from __future__ import absolute_import

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from config import cfg
from bbox_transform import get_out_utt_boxes, bbox_overlaps, bbox_transform, bbox_transform_batch

try:
    long # python 2
except NameError:
    long = int # python 3  

class AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produce anchor classification 
        labels and bounding-box regression targets
    """
    def __init__(self):
        super(AnchorTargetLayer, self).__init__()


    def forward(self, anchors, gt_boxes, act_lens):
        # here the anchors should be the anchors for each utterance, because 
        # when we calculate the training target (before RPN )  for each
        # utterance, the anchors are exactly the same (different from proposals)
        batch_size = gt_boxes.size(0)
        num_anchors_per_utt = anchors.size(0)
        rpn_labels = gt_boxes.new(batch_size, anchors.size(0)).fill_(-1)
        bbox_inside_weights = gt_boxes.new(batch_size, anchors.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, anchors.size(0)).zero_()
        
        overlaps = bbox_overlaps(anchors, gt_boxes[:,:, 1:]) 
        # batch_size * num_anchors_per_utt * num_gt_boxes
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        # For each anchor, we will find a max gt_boxes as its 
        # potential training target

        # fg label
        rpn_labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        # bg label 
        rpn_labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP ] = 0

        # disable the anchors out of utterance
        disable_indexes = get_out_utt_boxes(anchors, act_lens, batch_size)
        rpn_labels[disable_indexes] = -1

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        sum_fg = torch.sum((rpn_labels == 1).int(), 1)
        sum_bg = torch.sum((rpn_labels == 0).int(), 1)

        for i in range(batch_size):
            # subsample positive labels if we have too man
            fg_inds = torch.nonzero(rpn_labels[i] == 1).view(-1)
            bg_inds = torch.nonzero(rpn_labels[i] == 0).view(-1)
            if fg_inds.size(0) > 0:
                rpn_labels[i][fg_inds] = torch.index_select(gt_boxes[i], 0, argmax_overlaps[i][fg_inds])[:, 0]
            if sum_fg[i] > num_fg:
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                rpn_labels[i][disable_inds] = -1


            num_bg = int(cfg.TRAIN.RPN_BATCHSIZE - torch.sum((rpn_labels[i] == 1).int()))
            if sum_bg[i] > num_bg:
                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]] 
                rpn_labels[i][disable_inds] = -1

        bbox_inside_weights[rpn_labels > 0] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS
        bbox_inside_weights = bbox_inside_weights.view(batch_size, num_anchors_per_utt, 1).expand(batch_size, num_anchors_per_utt, 2)
        num_positive = torch.sum(rpn_labels > 0)
        num_negative = torch.sum(rpn_labels == 0)
        if cfg.DEBUG:
            print('Num positive samples: {}, num negative samples: {}'.format(num_positive, num_negative))
        if num_positive < 1:
            num_positive += 1
        positive_weights = 1.0 / num_positive.item()
        negative_weights = 1.0 / num_positive.item()
        bbox_outside_weights[rpn_labels > 0] = positive_weights
        bbox_outside_weights[rpn_labels == 0] = negative_weights
        bbox_outside_weights = bbox_outside_weights.view(batch_size, num_anchors_per_utt, 1).expand(batch_size, num_anchors_per_utt,2)
        # compute bbox regression target of anchors
        
        # here for each utterance in the batch, we only choose the best matching 
        # gt_box to calculate the bbox_targets for each anchor
        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        rpn_targets = bbox_transform_batch(anchors, gt_boxes[:,:,1:].view(-1,2)[argmax_overlaps.view(-1), :].view(batch_size, -1, 2)) # num_anchors * num_gt_boxes
        return rpn_labels, rpn_targets, bbox_inside_weights, bbox_outside_weights
