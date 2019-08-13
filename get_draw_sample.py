import os
import sys
import shutil

import add_path
from build_scp import build_scp_dict

if __name__=="__main__":
    if len(sys.argv) < 4:
        print("UDAGE:python %s wav.scp output_dir utt_id.list\n"%(sys.argv[0]))
        exit(1)
    wav_dict = build_scp_dict(sys.argv[1])
    output_dir = sys.argv[2]

    for utt_id in open(sys.argv[3]).readlines():
        source_file=wav_dict[utt_id.strip()]
        target_file=output_dir + "/" + utt_id.strip() + ".wav"
        shutil.copy(source_file, target_file)
