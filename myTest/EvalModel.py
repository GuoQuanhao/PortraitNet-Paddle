import paddle
import paddle.nn as nn
from paddle.io import DataLoader

import os
import numpy as np
from easydict import EasyDict as edict
from yaml import load

import sys
sys.path.append('../data/')
sys.path.append('../util/')
sys.path.append('../model/')

from datasets import Human


__Author__ = 'Quanhao Guo'
__Date__ = '2021.05.03.16.29'


def calcIOU(img, mask):
    sum1 = img + mask
    sum1[sum1>0] = 1
    sum2 = img + mask
    sum2[sum2<2] = 0
    sum2[sum2>=2] = 1
    if np.sum(sum1) == 0:
        return 1
    else:
        return 1.0*np.sum(sum2)/np.sum(sum1)

def test(dataLoader, netmodel, exp_args):
    # switch to eval mode
    netmodel.eval()
    softmax = nn.Softmax(axis=1)
    iou = 0
    for i, (input_ori, input, edge, mask) in enumerate(dataLoader):  
        input_ori_var = input_ori
        mask_var = mask
        
        # compute output: loss part1
        if exp_args.addEdge == True:
            output_mask, output_edge = netmodel(input_ori_var)
        else:
            output_mask = netmodel(input_ori_var)
            
        prob = softmax(output_mask)[0,1,:,:]
        pred = prob.numpy()
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0
        iou += calcIOU(pred, mask_var[0].numpy())
        
    print(len(dataLoader))
    return iou/len(dataLoader)

# load model-1 or model-2: trained with two auxiliary losses (without prior channel)
config_path = '../config/model_mobilenetv2_with_two_auxiliary_losses.yaml'

# load model-3: trained with prior channel 
# config_path = '../config/model_mobilenetv2_with_prior_channel.yaml'

with open(config_path,'rb') as f:
    cont = f.read()
cf = load(cont)

print('finish load config file ...')
print('===========> loading data <===========')
exp_args = edict()    
exp_args.istrain = False
exp_args.task = cf['task']
exp_args.datasetlist = cf['datasetlist'] # ['EG1800', ATR', 'MscocoBackground', 'supervisely_face_easy']
print("datasetlist: ", exp_args.datasetlist)

exp_args.model_root = cf['model_root'] 
exp_args.data_root = cf['data_root']
exp_args.file_root = cf['file_root']

# the height of input images, default=224
exp_args.input_height = cf['input_height']
# the width of input images, default=224
exp_args.input_width = cf['input_width']

# if exp_args.video=True, add prior channel for input images, default=False
exp_args.video = cf['video']
# the probability to set empty prior channel, default=0.5
exp_args.prior_prob = cf['prior_prob']

# whether to add boundary auxiliary loss, default=False
exp_args.addEdge = cf['addEdge']
# whether to add consistency constraint loss, default=False
exp_args.stability = cf['stability']

# input normalization parameters
exp_args.padding_color = cf['padding_color']
exp_args.img_scale = cf['img_scale']
# BGR order, image mean, default=[103.94, 116.78, 123.68]
exp_args.img_mean = cf['img_mean']
# BGR order, image val, default=[1/0.017, 1/0.017, 1/0.017]
exp_args.img_val = cf['img_val'] 

# if exp_args.useUpsample==True, use nn.Upsample in decoder, else use nn.ConvTranspose2d
exp_args.useUpsample = cf['useUpsample'] 
# if exp_args.useDeconvGroup==True, set groups=input_channel in nn.ConvTranspose2d
exp_args.useDeconvGroup = cf['useDeconvGroup'] 

exp_args.init = False
exp_args.resume = True

dataset_test = Human(exp_args)
print(len(dataset_test))
dataLoader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)
print(len(dataLoader_test))
print("finish load dataset ...")

print('===========> loading model <===========')
import model_mobilenetv2_seg_small as modellib
netmodel = modellib.MobileNetV2(n_class=2, 
                                useUpsample=exp_args.useUpsample, 
                                useDeconvGroup=exp_args.useDeconvGroup, 
                                addEdge=exp_args.addEdge, 
                                channelRatio=1.0, 
                                minChannel=16, 
                                weightInit=True,
                                video=exp_args.video)

if exp_args.resume:
    bestModelFile = os.path.join(exp_args.model_root, 'mobilenetv2_eg1800_with_two_auxiliary_losses.pdparams')
    if os.path.isfile(bestModelFile):
        checkpoint = paddle.load(bestModelFile)
        netmodel.set_state_dict(checkpoint)
        print("=> loaded checkpoint '{}'".format(bestModelFile))
    else:
        print("=> no checkpoint found at '{}'".format(bestModelFile))
netmodel = netmodel

acc = test(dataLoader_test, netmodel, exp_args)
print("mean iou: ", acc)
