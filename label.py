#!usr/bin/env python
#
# Copyright 2018 houjingyong@gmail.com
#
# MIT Lisence


from __future__ import absolute_import
from __future__ import print_function

class KWS_Label():
    def __init__(self, meta_file):
        self.label_dict = dict()

        with open(meta_file) as fid:
            self.metadata = [line.strip().split() for line in fid ]
        for i in range(len(self.metadata)):
            key = self.metadata[i][0]
            self.label_dict[key] = []
            for j in range(1, len(self.metadata[i])-2, 2):
                self.label_dict[key].append([int(self.matadata[i][j]), # first position
                                             int(self.matadata[i][j+1]), # second position
                                             int(self.matadata[i][j+2])]) # class number of keyword

    def get_item(self, utt_id):
        if self.label_dict.has_key(utt_id):
            return self.label_dict[utt_id]
        else:
            return None


