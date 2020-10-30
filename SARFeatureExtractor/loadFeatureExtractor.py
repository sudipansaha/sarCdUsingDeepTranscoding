# -*- coding: utf-8 -*-
"""
Spyder Editor

This code loads the feature extractor corresponding to convolutional layer 3 and residual block 1.
Further details about those layers can be found in the Table I in the paper
Building Change Detection in VHR SAR Images via Unsupervised Deep Transcoding.
One of these layers (or both of them) can be used as deep feature extractor, in change detection or other applications..
FC layers can be appended with one of them and further trained for SAR related classification tasks.
Please note that we suggest to normalize the input image and transform them into log domain.

The feature dimension from both the layers shown here are 256
"""
import os
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from architectures import FeatureExtractorConvLayer3
from architectures import FeatureExtractorResidualBlock1

manualSeed=40
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)


#Initilizing net / model
input_nc=1 #input number of channels
output_nc=1 #output number of channels
ngf=64 # number of gen filters in first conv layer
norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
use_dropout=False
#device = pytorch.device('cuda')
netForFeatureExtractionConvLayer3=FeatureExtractorConvLayer3(input_nc, output_nc, ngf, norm_layer, use_dropout, 9)
netForFeatureExtractionResidualBlock1=FeatureExtractorResidualBlock1(input_nc, output_nc, ngf, norm_layer, use_dropout, 9)


netForFeatureExtractionResidualBlock1Dict=netForFeatureExtractionResidualBlock1.state_dict()
state_dictForResidualBlock1=torch.load('./trainedModels/sarFeatureExtractorUptoResidualBlock1.pth')

netForFeatureExtractionConvLayer3Dict=netForFeatureExtractionConvLayer3.state_dict()
state_dictForConvLayer3=torch.load('./trainedModels/sarFeatureExtractorUptoConvLayer3.pth')


netForFeatureExtractionResidualBlock1.load_state_dict(state_dictForResidualBlock1)
netForFeatureExtractionConvLayer3.load_state_dict(state_dictForConvLayer3)


print(netForFeatureExtractionResidualBlock1)
print(netForFeatureExtractionConvLayer3)


##changing all nets to eval mode (for feature extraction)
#netForFeatureExtractionResidualBlock1.eval()
#netForFeatureExtractionConvLayer3.eval()


