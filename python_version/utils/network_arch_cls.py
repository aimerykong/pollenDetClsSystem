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



class ResnetBase(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained):
        super(ResnetBase, self).__init__()
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
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.layer1(self.encoder.maxpool(x))
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)        
        x = self.encoder.layer4(x)

        return x

    
    
class AlexBase(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexBase, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))        

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x
    
    
    
    
    
    
class PollenClsNet(nn.Module):
    def __init__(self, num_layers, nClass=48, modelName='resnet', pretrained=False, poolSize=4):
        super(PollenClsNet, self).__init__()
        if modelName=='resnet':
            self.encoder = ResnetBase(num_layers, pretrained)
        elif 'alex' in modelName:
            self.encoder = AlexBase(num_classes=nClass)
        else:
            self.encoder = AlexBase(num_classes=nClass)
            
        self.pool = nn.MaxPool2d(poolSize, poolSize)
        if num_layers>34:
            self.fc = nn.Linear(2048, nClass)
        else:
            self.fc = nn.Linear(512, nClass)
        
        
    def forward(self, x):  
        x = (x )
        x = self.encoder(x)
        x = self.pool(x)
        x = x.view(-1, x.shape[1])
        x = self.fc(x)
        return x