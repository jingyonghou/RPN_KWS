#!/usr/bin/env python

# Copyrigh 2018 houjingyong@gmail.com

# MIT Licence

import torch
import torch.nn as nn
import torch.nn.functional as F

from basic_nodes import LinearBlock

class FinalLayer(nn.Module):
    """
    final classification and bounding box regression layer for RPN KWS
    """
    def __init__(self, input_dim, num_class):
        super(FinalLayer, self).__init__()
        self.linear = LinearBlock(input_dim, input_dim, activation="relu")
        self.cls_score_KWS = nn.Linear(input_dim, num_class, bias=True)
        self.bbox_score_KWS = nn.Linear(input_dim, 2, bias=True)

    def forward(self, x):
        x = self.linear(x)
        kws_cls_score = self.cls_score_KWS(x)
        kws_bbox_pred = self.bbox_score_KWS(x)

        return kws_cls_score, kws_bbox_pred
