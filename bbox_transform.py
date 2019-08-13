# # Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# 
# Modified by Jingyong Hou to suit to one dimention case, 2019

import torch
import numpy as np

def bbox_transform(ex_rois, gt_rois):
    ex_length = ex_rois[:, :,1::2] - ex_rois[:,:,0::2] + 1.0
    ex_ctr = ex_rois[:,:,0::2] + 0.5 * ex_length

    gt_length = gt_rois[:,:,1::2] - gt_rois[:,:,0::2] + 1.0
    gt_ctr = gt_rois[:,:,0::2] + 0.5 * gt_length

    targets_dc = (gt_ctr - ex_ctr) / ex_length
    targets_dl = torch.log(gt_length / ex_length)

    return torch.cat([targets_dc, targets_dl], dim=2)

def bbox_transform_batch(ex_rois, gt_rois):
    if ex_rois.dim() == 2:
        ex_length = ex_rois[:, 1::2] - ex_rois[:, 0::2] + 1.0 
        # num_rois * 1
        ex_ctr = ex_rois[:, 0::2] + 0.5 * ex_length

        gt_length = gt_rois[:,:,1::2] - gt_rois[:,:,0::2] + 1.0
        # batch_size * num_rois * 1
        gt_ctr = gt_rois[:,:,0::2] + 0.5 * gt_length

        targets_dc = (gt_ctr - ex_ctr.view(1, -1, 1).expand_as(gt_ctr)) / ex_length.view(1,-1,1).expand_as(gt_ctr)
        targets_dl = torch.log(gt_length / ex_length.view(1, -1, 1).expand_as(gt_length))
    else:
        return bbox_transform(ex_rois, gt_rois)
    return torch.cat([targets_dc, targets_dl], dim=2)

def bbox_transform_inv(boxes, deltas):
    lengths = boxes[:,:,1::2] - boxes[:,:,0::2] + 1.0
    ctr = boxes[:,:,0::2] + 0.5 * lengths
    
    dc = deltas[:,:,0::2]
    dl = deltas[:,:,1::2]

    pred_l = lengths * torch.exp(dl)  
    pred_ctr = dc * lengths + ctr

    pred_x1 = pred_ctr - 0.5 * pred_l
    pred_x2 = pred_ctr + 0.5 * pred_l

    return torch.cat([pred_x1, pred_x2], dim=2)

def clip_boxes(boxes, act_lens, batch_size):
    for i in range(batch_size):
        boxes[i, :, 0::2].clamp_(0, act_lens[i]-1)
        boxes[i, :, 1::2].clamp_(0, act_lens[i]-1)
    return boxes

def get_small_big_boxes(boxes, min_box=30, max_box=220):
    lengths = boxes[:, :, 1] - boxes[:, :, 0] + 1.0
    keep = ((lengths >= min_box) & (lengths <= max_box))
    inds_remove = (keep == 0)
    return inds_remove

def get_out_utt_boxes(boxes, act_lens, batch_size):
    keep = torch.zeros(batch_size, boxes.size(0)).type_as(boxes)
    for i in range(batch_size):
        keep[i,:] = ((boxes[:, 0] >= 0) & (boxes[:, 1] < act_lens[i].float()))
    inds_outside = (keep < 0.5)
    return inds_outside

def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 2) ndarray of float
    gt_boxes: (batch_size, K, 2) ndarray of float
    
    overlaps: (batch_size, N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    batch_size = gt_boxes.size(0)
    K = gt_boxes.size(1)

    anchors_length = (anchors[:,1] - anchors[:,0] + 1).view(1, N,1).expand(batch_size, N, 1)
    gt_boxes_length = (gt_boxes[:,:,1] - gt_boxes[:,:,0] + 1).view(batch_size, 1, K)
    boxes = anchors.view(1, N, 1, 2).expand(batch_size, N, K, 2)
    query_boxes = gt_boxes.view(batch_size, 1, K, 2).expand(batch_size, N, K, 2)
    inter_length = (torch.min(boxes[:,:,:,1], query_boxes[:,:,:,1]) - torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0])).view(batch_size, N, K)
    inter_length[inter_length < 0] = 0
    union_length = (anchors_length + gt_boxes_length) - inter_length      
    overlaps = inter_length/union_length
    return overlaps

def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (batch_size, N, 2) ndarray of float
    gt_boxes: (batch_size, K, 2) ndarray of float
    
    overlaps: (batch_size, N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(1)
    batch_size = gt_boxes.size(0)
    K = gt_boxes.size(1)

    anchors_length = (anchors[:,:,1] - anchors[:,:,0] + 1).view(batch_size, N, 1)
    gt_boxes_length = (gt_boxes[:,:,1] - gt_boxes[:,:,0] + 1).view(batch_size, 1, K)
    boxes = anchors.view(batch_size, N, 1, 2).expand(batch_size, N, K, 2)
    query_boxes = gt_boxes.view(batch_size, 1, K, 2).expand(batch_size, N, K, 2)
    inter_length = (torch.min(boxes[:,:,:,1], query_boxes[:,:,:,1]) - torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0])).view(batch_size, N, K)
    inter_length[inter_length < 0] = 0
    union_length = (anchors_length + gt_boxes_length) - inter_length      
    overlaps = inter_length/union_length
    return overlaps
