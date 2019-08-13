from __future__ import print_function

import torch
import numpy as np
import pdb

try:
    xrange
except:
    xrange = range # python3

class AnchorGenerator:
    def __init__(self, num_anchors_per_frame=20, min_box_size=30, max_box_size=220, max_utt_length=1500, device="cuda"):
        self.num_anchors_per_frame = num_anchors_per_frame
        self.max_box_size = max_box_size
        self.min_box_size = min_box_size
        self.max_utt_length = max_utt_length
        x = self._generate_basic_anchors(num_anchors_per_frame, min_box_size, max_box_size)
        self.basic_anchors = torch.from_numpy(x).float()
        self._update(max_utt_length, device)

    def _generate_basic_anchors(self, num_anchors_per_frame, min_window_size, max_window_size):
        shift = (max_window_size-min_window_size)/1.0/(num_anchors_per_frame-1)
        start_indexes = np.arange(min_window_size, max_window_size+1, shift)
        basic_anchors = np.zeros([num_anchors_per_frame, 2])
        basic_anchors[:,0]=-start_indexes
        basic_anchors[:,1]=0
        return basic_anchors

    def _generate_log_anchors(self, num_anchors_per_frame, min_window_size, max_window_size):
        log_min = np.log(min_window_size)
        log_max = np.log(max_window_size)
        log_shift = (log_max-log_min)/1.0/(num_anchors_per_frame-1)
        log_start_indexes = np.arange(log_min, log_max+log_shift, log_shift)
        start_indexes = np.exp(log_start_indexes)
        basic_anchors = np.zeros([num_anchors_per_frame, 2])
        basic_anchors[:,0]=-start_indexes
        basic_anchors[:,1]=0
        return basic_anchors

    def _update(self, max_utt_length, device):
        self.device = torch.device(device)
        self.max_utt_length = max_utt_length
        shifts = torch.arange(0, max_utt_length).float()
        self.anchors_per_utt = self.basic_anchors.view(1, self.num_anchors_per_frame, 2) +shifts.view(max_utt_length, 1, 1).expand(max_utt_length, 1, 2)
        self.anchors_per_utt = self.anchors_per_utt.to(self.device)

    def get_anchors_per_utt(self, length, device="cuda"):
        if length < self.max_utt_length:
            return self.anchors_per_utt[0:length,:]
        else:
            self._update(length, device)
            return self.anchors_per_utt
