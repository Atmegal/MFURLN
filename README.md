# MFURLN

###Pytorch code for our CVPR 2019 paper ["On Exploring Undetermined Relationships for Visual Relationship Detection."](https://arxiv.org/pdf/1905.01595.pdf)

##Introduction

This implementation is based on [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0).

So you can reference this project to build you own code environmnt and compile the cuda dependencies.

##Prerequisites

  *Python 3.6
  *Pytorch 1.0 
  *CUDA9.0

## Train

### Train relationship detection model:
```
python train_rela.py --datasets VRD --net vgg16 --lr 0.005 --lr_decay_step 1 --lr_decay_gamma 0.5 --cuda
```

## Test
### Test relationship detection model:
```
python test_rela.py --datasets VRD --net vgg16 --cuda
```
