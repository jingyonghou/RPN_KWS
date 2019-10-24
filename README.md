# RPN_KWS
Pytorch code of paper "Region Proposal Network Based Small-Footprint Keyword Spotting"
https://ieeexplore.ieee.org/document/8807313

Please cite the work below if you want to use the code or want to do research related to our work
```
@ARTICLE{hou2019region, 
author={Hou, Jingyong and Shi, Yangyang and Ostendorf, Mari and  Hwang, Mei-Yuh
and Xie, Lei }, 
journal={IEEE Signal Processing Letters}, 
title={Region Proposal Network Based Small-Footprint Keyword Spotting}, 
year={2019}, 
volume={26}, 
number={10}, 
pages={1471-1475}
}
```

I will release a new version of RPN KWS with an Online Hard Example Mining (OHEM) algorithm, which will improve our system.

https://github.com/jingyonghou/RPN_KWS_OHEM

## Detection samples
![image](https://github.com/jingyonghou/RPN_KWS/raw/master/Picture1.png)

![image](https://github.com/jingyonghou/RPN_KWS/raw/master/Picture2.png)

Selected two utterances which contains predefined keyword. The red box is the ground-truth start-end area of keyword from forced-alignment, the blue box is the best anchor selected according to the classification score, the green box is the proposed region proposal corresponding to the best anchor.


## Running environment
### Python 2.7.15
### pytorch 0.4.1
### CUDA 8.0 or higher
### Kaldi
You should know basic knowledge of Kaldi before looking at the run script. I use Kaldi to extract Fbank features and do a global CMVN using the statictics from all training set. You should add cmd.sh, path.sh, steps and utils to your working dir before you run the script.

### Please follow the run_rpn_kws.sh script to learn how to run the code

## reference
https://github.com/jwyang/faster-rcnn.pytorch

https://github.com/vesis84/kaldi-io-for-python
