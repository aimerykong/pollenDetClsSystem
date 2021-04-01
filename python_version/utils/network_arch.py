from __future__ import absolute_import, division, print_function
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
from collections import OrderedDict
from utils.layers import *
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import numpy as np
import os, math
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn



class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained):
        super(ResnetEncoder, self).__init__()
        self.path_to_model = './model'
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 
                   101: models.resnet101, 152: models.resnet152}
        resnets_pretrained_path = {18: 'resnet18-5c106cde.pth', 
                                   34: 'resnet34.pth', 50: 'resnet50.pth', 101: 'resnet101.pth', 152: 'resnet152.pth'}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(
                num_layers))

        self.encoder = resnets[num_layers]()

        if pretrained:
            print("using pretrained model")
            self.encoder.load_state_dict(
                torch.load(os.path.join(self.path_to_model, resnets_pretrained_path[num_layers])))
        
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []

        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(
            self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
    

    
    
class Decoder(nn.Module):
    def __init__(self, num_ch_enc, scaleList=(0, 1, 2, 3),
            num_output_channels=1, use_skips=True):
        super(Decoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scaleList = scaleList

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scaleList:
            self.convs[("segMask", s)] = Conv3x3(self.num_ch_dec[s], 1)
            self.convs[("output", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            #self.convs[("outputAuxiliary", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scaleList:       
                self.outputs[("segMask", i)] = self.sigmoid(self.convs[("segMask", i)](x))
                self.outputs[("output", i)] = self.convs[("output", i)](x)

        return self.outputs



class PollenDet_SegDistTransform(nn.Module):
    def __init__(self, num_layers, scaleList=(0,1,2,3), pretrained=False):
        super(PollenDet_SegDistTransform, self).__init__()
        self.scaleList = scaleList 
        self.encoder = ResnetEncoder(num_layers, pretrained)
        self.decoder = Decoder(self.encoder.num_ch_enc, scaleList=self.scaleList, num_output_channels=1)
        
    def forward(self, inputs1):  
        inputs = (inputs1 )
        features = self.encoder(inputs)
        outputs = self.decoder(features)
        return outputs
    
    
class PollenDet_SegKeyRadius(nn.Module):
    def __init__(self, num_layers, scaleList=(0,1,2,3), pretrained=False):
        super(PollenDet_SegKeyRadius, self).__init__()
        self.scaleList = scaleList 
        self.encoder = ResnetEncoder(num_layers, pretrained)
        self.decoder = Decoder(self.encoder.num_ch_enc, scaleList=self.scaleList, num_output_channels=2)
        
    def forward(self, inputs1):  
        inputs = (inputs1 )
        features = self.encoder(inputs)
        outputs = self.decoder(features)
        return outputs
    
    
class PollenDet_SegVoting(nn.Module):
    def __init__(self, num_layers, scaleList=(0,1,2,3), pretrained=False):
        super(PollenDet_SegVoting, self).__init__()
        self.scaleList = scaleList 
        self.encoder = ResnetEncoder(num_layers, pretrained)
        self.decoder = Decoder(self.encoder.num_ch_enc, scaleList=self.scaleList, num_output_channels=2)
        
    def forward(self, inputs1):  
        inputs = (inputs1 )
        features = self.encoder(inputs)
        outputs = self.decoder(features)
        return outputs