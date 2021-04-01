import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
from scipy import misc
import matplotlib.pyplot as plt
import PIL.Image
import pickle

import skimage.transform 

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms





    
class PollenCls(Dataset):
    def __init__(self, dbinfo, size=[128, 128], set_name='train', maskType='pred'):
        if set_name=='val':
            set_name = 'test'
        self.set_name = set_name
        self.imageList = dbinfo[set_name+'_cls_imgList']
        
        if maskType=='pred':
            self.segList = dbinfo[set_name+'_cls_predMaskList']
        else:
            self.segList = dbinfo[set_name+'_cls_grndMaskList']
        
        self.clsIDList = dbinfo[set_name+'_cls_classID']
        self.transform = transform                
        self.size = size
        
        self.TFNormalize = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        self.current_set_len = len(self.imageList)
        
        self.TF2tensor = transforms.ToTensor()
        self.TF2PIL = transforms.ToPILImage()
        self.TFresize = transforms.Resize((self.size[0],self.size[1]))
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):
        curImageName = self.imageList[idx]
        curMaskName = self.segList[idx]
        curLabelIndex = self.clsIDList[idx].astype(np.float32)-1
        curLabelIndex = np.array([curLabelIndex]).astype(np.float32)
        
        image = PIL.Image.open(curImageName)
        segMask = PIL.Image.open(curMaskName)
        
        if self.set_name=='train' and np.random.random(1)>0.5:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            segMask = segMask.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            
        image = np.array(image)
        segMask = np.array(segMask).astype(np.float32)
        times = np.random.randint(4)
        image = np.rot90(image, times).copy()
        segMask = np.rot90(segMask, times).copy()
        
        #print('before: ', image.max(), image.min())
        
        #print('rot90 ', image.max(), image.min())
        
        
        sz = image.shape[:2]
        newSz = self.size[0]
        if newSz<sz[0] or newSz<sz[1]:
            yNew = (sz[0]-newSz)//2
            yNew = max(yNew,0)
            xNew = (sz[1]-newSz)//2
            xNew = max(xNew,0)
            
            image = image[yNew:yNew+newSz, xNew:xNew+newSz, :].copy()
            segMask = segMask[yNew:yNew+newSz, xNew:xNew+newSz].copy()
            #print('if: ', image.max(), image.min())
        else:
            yNew = (newSz-sz[0])//2
            yNew = max(yNew, 0)
            xNew = (newSz-sz[1])//2
            xNew = max(xNew, 0)
            
            imageNew = np.zeros((self.size[0],self.size[1], 3), dtype=np.uint8)
            segMaskNew = np.zeros((self.size[0],self.size[1]), dtype=np.float32)
            imageNew[yNew:yNew+sz[0], xNew:xNew+sz[1], :] = image
            segMaskNew[yNew:yNew+sz[0], xNew:xNew+sz[1]] = segMask            
            image = imageNew.copy()
            segMask = segMaskNew.copy()        
            #print('else: ', image.max(), image.min())
        
        image = self.TF2tensor(image)
        #print('to tensor ', image.max(), image.min())
        
        image = self.TFNormalize(image)
        #print('normalize ', image.max(), image.min())
        
        segMask = torch.from_numpy(segMask).unsqueeze(0) # self.TF2tensor(mask_overlap)                
        label = torch.from_numpy(curLabelIndex).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        image = image.unsqueeze(0)
        segMask = segMask.unsqueeze(0)        
        
        #print(image.shape, segMask.shape)
        image = F.interpolate(image, size=(self.size[0],self.size[1]), mode='bilinear', align_corners=True)
        segMask = F.interpolate(segMask, size=(self.size[0], self.size[1]), mode='nearest')
        image = image.squeeze(0)        
        segMask = segMask.squeeze(0)    
        label = label.squeeze(0)
        return image, segMask, label    