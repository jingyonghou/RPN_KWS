from __future__ import print_function
import os
import sys

import numpy as np
from kaldi_io import *

keyword_dict={"hixiaowen":1, 
            "nihaowenwen":2,
            "freetext":0,
            "1":1,
            }

keyword_list = ["hixiaowen", "nihaowenwen", "1"]

if __name__=="__main__":
    if len(sys.argv) < 4:
        print("USAGE:python %s output.ark keyword score.txt"%sys.argv[0])
        exit(1)
    keyword = sys.argv[2]
    fid = open(sys.argv[3],"w")
    
    ignore_keyword = False
    if len(sys.argv) == 5:
        ignore_keyword = bool(int(sys.argv[4]))

    for utt_id, mat in read_mat_ark(sys.argv[1]):
        keyword_id = utt_id.strip().split("_")[-1]
        if keyword_id == keyword:
            fid.writelines("1")
        elif (keyword_id in keyword_list) and ignore_keyword:
            continue
        else:
            fid.writelines("0")
        fid.writelines(" "+utt_id)
        for i in range(mat.shape[0]):
            fid.writelines(" %f"%mat[i, keyword_dict[keyword]])
        fid.writelines("\n")

    fid.close()

