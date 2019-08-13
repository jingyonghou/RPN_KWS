#!/usr/bin/env python

# Copyrigh 2018 houjingyong@gmail.com

# Apache 2.0.                                                                                                                                                   
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def loss_frame_fn_ce(output, target):
    return F.nll_loss(output, target.long())

def acc_frame(output, target):
    if output is None:
        return 0
    pred = output.max(1, keepdim=True)[1]
    correct = pred.eq(target.long().view_as(pred)).sum().item()
    return correct*100.0/output.size(0)

def loss_frame_fn_focal(output, target, gamma=0):
    target = target.view(-1, 1)
    logpt = output
    logpt = logpt.gather(1, target)
    logpt = logpt.view(-1)

    pt = Variable(logpt.data.exp())
    loss = -1 * (1-pt)**gamma * logpt
    return loss.mean()

def loss_frame_fn_focal_debug(output, target, gamma=0):
    idx_p = (target > 0)
    idx_n = (target == 0 )
    n_p = torch.sum(idx_p)
    n_n = torch.sum(idx_n)

    target = target.view(-1, 1)
#    print(target.shape)
#    print(logpt.shape)
    logpt = output
    logpt = logpt.gather(1, target)
    logpt = logpt.view(-1)

    logpt_p = logpt[idx_p]
    logpt_n = logpt[idx_n]
    pt_p = Variable(logpt_p.data.exp())
    pt_n = Variable(logpt_n.data.exp())
    pt = Variable(logpt.data.exp())
    loss_p = -1 * (1-pt_p)**gamma * logpt_p
    loss_n = -1 * (1-pt_n)**gamma * logpt_n
    #print("positive loss: %d, %f, negative loss: %d, %f\n"%(int(n_p),float(loss_p.mean()),int(n_n),float(loss_n.mean())))
    #print("positive loss:%d:%f, negative loss:%d:%f\n"%(int(n_p),float(loss_p.mean()),int(n_n),float(loss_n.mean())))
    loss = -1 * (1-pt)**gamma * logpt
    # TODO wrong 
    return loss.mean()
    
def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=100.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box




