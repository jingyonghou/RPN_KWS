# RPN_KWS
Pytorch code of paper "Region Proposal Network Based Small-Footprint Keyword Spotting"

```
@article{hou2019Region,
  title={Region Proposal Network Based Small-Footprint Keyword Spotting},
  author={Jingyong Hou and Yangyang Shi and Mari Ostendorf and Mei-Yuh Hwang and Lei Xie},
  journal={Signal Processing Letters (\textbf{accepted})},
  year={2019}
}
```

# Running enviroment
## Python 2.7.15
## pytorch 0.4.1
## CUDA 8.0 or higher
## Kaldi
You should know basic knowledge of Kaldi before look at the run script, I used Kaldi to extract Fbank features and did a globel CMVN use the statictics from all training set. You should add cmd.sh, path.sh, steps and utils to your working path before you run the script.

## Please follow the run_rpn_kws.sh script to learn how to run the code

## reference
https://github.com/jwyang/faster-rcnn.pytorch
https://github.com/vesis84/kaldi-io-for-python
