#!/usr/bin/env python

# Copyrigh 2018 houjingyong@gmail.com

# MIT Licence

import sys
import numpy as np

keyword_dict={"hixiaowen":1, 
            "nihaowenwen":2,
            "freetext":0
            }

keyword_list = ["hixiaowen", "nihaowenwen"]
def IoU(s1, e1, s2, e2):
    inter = (min(e1, e2)-max(s1, s2))
    if inter < 0:
        inter = 0
    union = e1-s1+e2-s2-inter
    return inter/union

def get_overlap_ratio(label_string, anchor_string, region_string, shift=0):
    gt_score, gt_s, gt_e = map(float, label_string.strip().split(" "))
    anchor_score, anchor_s, anchor_e = map(float, anchor_string.strip().split(" "))
    region_score, region_s, region_e = map(float, region_string.strip().split(" "))

    score = region_score
    anchor_ratio = IoU(gt_s, gt_e, anchor_s+shift, anchor_e+shift)
    region_ratio = IoU(gt_s, gt_e, region_s+shift, region_e+shift)

    return score, anchor_ratio, region_ratio
    

if __name__=="__main__":
    if len(sys.argv) < 5:
        print("USAGE:python %s region keyword outputfile\n"%(sys.argv[0]))
        exit(1)
    region_file = sys.argv[1]
    keyword = sys.argv[2]
    output_file = sys.argv[3]
    fid = open(output_file, "w")
    shift = float(sys.argv[4])
    # for each line calculate the overlap ratio
    for line in open(region_file).readlines():
        utt_id, label, anchor1, anchor2, region1, region2 = line.strip().split(",") 
        keyword_id = utt_id.strip().split("_")[-1]
        if keyword_id == keyword:
            if keyword_dict[keyword_id] == 1:
                score, anchor_ratio, region_ratio = get_overlap_ratio(label, anchor1, region1, shift)
            elif keyword_dict[keyword_id] == 2:
                score, anchor_ratio, region_ratio = get_overlap_ratio(label, anchor2, region2, shift)
            else:
                pass
            fid.writelines("%f %f %f\n"%(score, anchor_ratio, region_ratio))
        else:
            pass
