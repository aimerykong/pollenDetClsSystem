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





def train_model(model, dataloaders, lossFunc,
                optimizer, scheduler, num_epochs=50, work_dir='./', device='cpu', freqShow=40):
    
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
            countSmpl = 0.
            
            # Iterate over data.
            iterCount, sampleCount = 0, 0
            for sample in dataloaders[phase]:                
                image, segMask, label = sample
                
                image = image.to(device)
                segMask = segMask.to(device) 
                label = label.type(torch.long).view(-1).to(device)
                
                
                
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
                    #print(image.shape, outputs.shape, label.shape)
                    
                    loss = lossFunc(outputs, label)
                        
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # statistics  
                iterCount += 1
                sampleCount += label.size(0)
                running_loss += loss.item() * label.size(0)                
                print2screen_avgLoss = running_loss / sampleCount
                       
                del loss                
                
                if iterCount%freqShow==0:
                    print('\t{}/{} loss:{:.3f}'.
                          format(iterCount, len(dataloaders[phase]), print2screen_avgLoss))
                    fn = open(log_filename,'a')        
                    fn.write('\t{}/{} loss:{:.3f}\n'.
                             format( iterCount, len(dataloaders[phase]), print2screen_avgLoss))
                    fn.close()
                    
                    
            epoch_loss = print2screen_avgLoss
                    
            print('\tloss: {:.6f}'.format(epoch_loss))
            fn = open(log_filename,'a')
            fn.write('\tloss: {:.6f}\n'.format(epoch_loss))
            fn.close()
            
            # deep copy the model            
            path_to_save_paramOnly = os.path.join(work_dir, 'epoch-{}.paramOnly'.format(epoch+1))
            torch.save(model.state_dict(), path_to_save_paramOnly)
            
            if (phase=='val' or phase=='test') and epoch_loss<best_loss:
                best_loss = epoch_loss

                path_to_save_paramOnly = os.path.join(work_dir, 'bestValModel.paramOnly')
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
