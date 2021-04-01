import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import models, transforms





def train_model(model, dataloaders, 
                lossFunc_seg, lossFunc_reg, 
                optimizer, scheduler,
                num_epochs=50, scaleList=[0,],
                work_dir='./', 
                device='cpu', freqShow=40):
    
    log_filename = os.path.join(work_dir, 'train.log')    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())    
    best_loss = float('inf')

    for epoch in range(num_epochs):        
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        fn = open(log_filename,'a')
        fn.write('\nEpoch {}/{}\n'.format(epoch+1, num_epochs))
        fn.write('--'*5+'\n')
        fn.close()


        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            print(phase)
            fn = open(log_filename,'a')        
            fn.write(phase+'\n')
            fn.close()
            
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode                
            else: 
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_loss_seg = 0.
            running_loss_reg = 0.
            countSmpl = 0.
            
            # Iterate over data.
            iterCount, sampleCount = 0, 0
            for sample in dataloaders[phase]:                
                image, grndSeg, grndDistTransform, mask_overlap, mask_voteX, mask_voteY, mask_peaks, mask_radius = sample
                
                image = image.to(device)
                grndSeg = grndSeg.to(device) 
                mask_peaks = mask_peaks.type(torch.FloatTensor).to(device)
                mask_radius = mask_radius.type(torch.FloatTensor).to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                loss = 0
                with torch.set_grad_enabled(phase=='train'):
                    if phase=='train':  
                        model.train()  # Set model to training mode                      
                    else:
                        model.eval()   # Set model to evaluate mode
                        
                    outputs = model(image)
                    
                    for ii in range(len(scaleList)):
                        #s_lab = outputScaleIndices[ii]
                        #s_img = outputImgScaleList[ii]
                        
                        #labels = sample[('normal', 0, s_img)].to(device)
                        #labelMask = sample[('normalMask', 0, s_img)].to(device)
                        #if s_lab == finestScale:
                        #    loss_fineScale = lossFunc(outputs[('output', s_lab)], labels, labelMask) / (2 ** s_lab)
                        #    loss += loss_fineScale
                        #else:
                        #    loss += lossFunc(outputs[('output', s_lab)], labels, labelMask) / (2 ** s_lab)
                        predSeg = outputs[('segMask', 0)]
                        predKeyRadius = outputs[('output', 0)]
                        grndKeyRadius = torch.cat((mask_peaks, mask_radius), 1)
                        
                        loss_seg = 0
                        loss_reg = 0
                        loss_seg = lossFunc_seg(predSeg, grndSeg)
                        
                        loss_reg = lossFunc_reg(grndKeyRadius, predKeyRadius, grndSeg)                        
                        loss = loss_seg + loss_reg
                        
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # statistics  
                iterCount += 1
                sampleCount += grndSeg.size(0)
                running_loss_seg += loss_seg.item() * grndSeg.size(0)
                running_loss_reg += loss_reg.item() * grndSeg.size(0)
                
                print2screen_avgLoss_seg = running_loss_seg / sampleCount
                print2screen_avgLoss_reg = running_loss_reg / sampleCount
                       
                del loss                
                
                if iterCount%freqShow==0:
                    print('\t{}/{} seg:{:.3f}, reg:{:.3f}'.
                          format(iterCount, len(dataloaders[phase]), 
                                 print2screen_avgLoss_seg, 
                                 print2screen_avgLoss_reg)
                         )
                    fn = open(log_filename,'a')        
                    fn.write('\t{}/{} l:{:.3f}, l-fine:{:.3f}\n'.
                             format( iterCount, len(dataloaders[phase]), 
                                    print2screen_avgLoss_seg, 
                                    print2screen_avgLoss_reg)
                            )
                    fn.close()
                    
                    
            epoch_loss = print2screen_avgLoss_seg + print2screen_avgLoss_reg
                    
            print('\tloss: {:.6f}'.format(epoch_loss))
            fn = open(log_filename,'a')
            fn.write('\tloss: {:.6f}\n'.format(epoch_loss))
            fn.close()
            
            # deep copy the model            
            path_to_save_paramOnly = os.path.join(work_dir, 'epoch-{}_encoder.paramOnly'.format(epoch+1))
            torch.save(model.state_dict(), path_to_save_paramOnly)
            
            if (phase=='val' or phase=='test') and epoch_loss<best_loss:
                best_loss = epoch_loss

                path_to_save_paramOnly = os.path.join(work_dir, 'bestValModel_encoder.paramOnly')
                torch.save(model.state_dict(), path_to_save_paramOnly)
                
                file_to_note_bestModel = os.path.join(work_dir,'note_bestModel.log')
                fn = open(file_to_note_bestModel,'a')
                fn.write('The best model is achieved at epoch-{}: loss{:.6f}.\n'.format(epoch+1,best_loss))
                fn.close()
                
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    fn = open(log_filename,'a')
    fn.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    fn.close()
   
    # load best model weights    
    model = model.load_state_dict(best_model_wts)
    
    return model
