# RPN_KWS
Pytorch code of paper "Region Proposal Network Based Small-Footprint Keyword Spotting"

Please cite the work below if you want to use the code or want to do research related to our work
```
@article{hou2019Region,
  title={Region Proposal Network Based Small-Footprint Keyword Spotting},
  author={Jingyong Hou and Yangyang Shi and Mari Ostendorf and Mei-Yuh Hwang and Lei Xie},
  journal={Signal Processing Letters (\textbf{accepted})},
  year={2019}
}
```
# Detection samples
![image](https://github.com/jingyonghou/RPN_KWS/raw/master/Picture1.png)

![image](https://github.com/jingyonghou/RPN_KWS/raw/master/Picture2.png)

The red box is the ground-truth box of keyword from force alignment, the blue box is best anchor selected according to the classification score, the green box is the proposed region proposal corresponding to the best anchor
## 

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
