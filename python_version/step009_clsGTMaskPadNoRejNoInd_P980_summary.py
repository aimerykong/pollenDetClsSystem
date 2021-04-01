#!/usr/bin/env python
# coding: utf-8

# import packages
# ------------------
# 
# Some packages are installed automatically if you use Anaconda. As pytorch is used here, you are expected to install that in your machine. 

# In[8]:


from __future__ import print_function, division
import os, random, time, copy
from sklearn.metrics import confusion_matrix
import seaborn as sn
from operator import itemgetter 
import pandas as pd
import numpy as np
import os.path as path
import scipy.io as sio
from scipy import misc
from scipy import ndimage, signal
import scipy
import pickle
import sys
import math
import matplotlib.pyplot as plt
import torch


# In[9]:


with open('dbinfo_cls_noReject_noInd.plk', 'rb') as handle:
    dbinfo = pickle.load(handle)

tmpval = -np.asarray(dbinfo['perClassCount'])
sort_index = np.argsort(tmpval)
top25indices = sort_index[0:25]
top25indices = torch.from_numpy(top25indices)
top25indices = top25indices.sort().values

    
plt.figure(figsize=(14, 5), dpi=200, facecolor='w', edgecolor='k') # figsize -- inch-by-inch

plt.bar(list(range(46)), dbinfo['perClassCount'])
plt.title('amount distribution per class')
plt.xlabel('class index')
plt.ylabel('number in train set')


plt.xticks(list(range(46)), dbinfo['meta'], size=10)
plt.xticks(rotation=45, ha='right')

#ax.tick_params(axis='both', which='major', labelsize=17)


# In[10]:


save_to_folder = './result' # experiment directory, used for reading the init model


filename = path.join(save_to_folder, 'step009_clsGTMaskPadNoRejNoInd_P001_res34.plk')    
with open(filename, 'rb') as handle:
    result_model1 = pickle.load(handle)
    
filename = path.join(save_to_folder, 'step009_clsGTMaskPadNoRejNoInd_P002_res34_wSqrtInvFreq.plk')    
with open(filename, 'rb') as handle:
    result_model2 = pickle.load(handle)


# all-47
# --

# In[11]:


accDict = {}

totalClassNum = len(dbinfo['perClassCount'])
A = result_model1['grndLabel_noReject']
B = result_model1['predLabel_noReject']
accDict['accPerClass_model1'] = [0.0]*totalClassNum
accPerClass_verify = [0.0]*totalClassNum
for i in range(A.shape[0]):
    curClsIdx = A[i]
    curClsIdx = int(curClsIdx.numpy())
    predClsIdxPred = B[i]
    predClsIdxPred = int(predClsIdxPred.numpy())
    accDict['accPerClass_model1'][curClsIdx] += float(curClsIdx==predClsIdxPred)
    accPerClass_verify[curClsIdx] += 1

for i in range(totalClassNum):
    accDict['accPerClass_model1'][i] = accDict['accPerClass_model1'][i] * 1.0 / (accPerClass_verify[i]+0.00001)
    
    
    
A = result_model2['grndLabel_noReject']
B = result_model2['predLabel_noReject']
# B = B[A!=47]
# A = A[A!=47]
accDict['accPerClass_model2'] = [0.0]*totalClassNum
accPerClass_verify = [0.0]*totalClassNum
for i in range(A.shape[0]):
    curClsIdx = A[i]
    curClsIdx = int(curClsIdx.numpy())
    predClsIdxPred = B[i]
    predClsIdxPred = int(predClsIdxPred.numpy())
    accDict['accPerClass_model2'][curClsIdx] += float(curClsIdx==predClsIdxPred)
    accPerClass_verify[curClsIdx] += 1

for i in range(totalClassNum):
    accDict['accPerClass_model2'][i] = accDict['accPerClass_model2'][i] * 1.0 / (accPerClass_verify[i]+0.00001)
      


# In[12]:


tmpval = -np.asarray(accPerClass_verify)
#sort_index = np.argsort(tmpval)
sort_index = torch.from_numpy(tmpval)
sort_index = sort_index.sort().indices

nameList = itemgetter(*sort_index.numpy().tolist())(dbinfo['meta'])

tmp_num = itemgetter(*sort_index.numpy().tolist())(accPerClass_verify)
tmp_acc1 = itemgetter(*sort_index.numpy().tolist())(accDict['accPerClass_model1'])
tmp_acc2 = itemgetter(*sort_index.numpy().tolist())(accDict['accPerClass_model2'])


x = np.arange(len(sort_index))  # the label locations
width = 0.4  # the width of the bars


#plt.figure(figsize=(8, 4), dpi=200, facecolor='w', edgecolor='k') # figsize -- inch-by-inch

fig, ax = plt.subplots(1, sharex=True, figsize=(18,5), dpi=320)
rects1 = ax.bar(x - width/2, tmp_acc1, width, label='w/o reweighting')
#rects2 = ax.bar(x + width/2, tmp_acc2, width, label='w/ reweighting')
rects2 = ax.bar(x + width/2, tmp_acc2, width, label='w/ reweighting')


ax.set_ylabel('accuracy', size=25)
ax.set_title('accuracy sorted by number of samples', size=25)
ax.set_xticks(x)
ax.set_xticklabels(nameList, size=15)
ax.legend(fontsize=17)
plt.xticks(rotation=45, ha='right')

ax.tick_params(axis='both', which='major', labelsize=17)


fig.tight_layout()
plt.show()


# In[13]:


fig, ax = plt.subplots(1, sharex=True, figsize=(18,5), dpi=320)
rects = ax.bar(x, tmp_num, width, label='w/o reweighting')

ax.set_ylabel('#samples', size=25)
ax.set_title('#samples per class', size=25)
ax.set_xticks(x)
ax.set_xticklabels(nameList, size=15)
ax.legend(fontsize=17)
plt.xticks(rotation=45, ha='right')

ax.tick_params(axis='both', which='major', labelsize=17)


fig.tight_layout()
plt.show()


# top-25
# --

# In[14]:


totalClassNum = 25

grndLabel_top25_model1 = result_model1['grndLabel_top25']
predLabel_top25_model1 = result_model1['predLabel_top25']
grndLabel_top25_model2 = result_model2['grndLabel_top25']
predLabel_top25_model2 = result_model2['predLabel_top25']

a = []
for i in range(len(grndLabel_top25_model1)):
    if int(grndLabel_top25_model1[i].item()) in top25indices:
        a += [i]

        
        
B_top25_model1 = predLabel_top25_model1[a]
A_top25_model1 = grndLabel_top25_model1[a]
B_top25_model2 = predLabel_top25_model2[a]
A_top25_model2 = grndLabel_top25_model2[a]



top25indicesList = top25indices.cpu().numpy().tolist()
for i in range(len(A_top25_model1)):
    elm = A_top25_model1[i]
    A_top25_model1[i] = top25indicesList.index(elm) 
    elm = A_top25_model2[i]
    A_top25_model2[i] = top25indicesList.index(elm)       

#print(A_top25.shape, B_top25.shape)
A_top25_model1 = A_top25_model1.numpy()
B_top25_model1 = B_top25_model1.numpy()
A_top25_model2 = A_top25_model2.numpy()
B_top25_model2 = B_top25_model2.numpy()

# accList = A_top25-B_top25
# accList = (accList==0).astype(np.float32)
# acc = accList.mean()
# print(acc)

nameList = itemgetter(*top25indices.numpy().tolist())(dbinfo['meta'])
numPerCls = itemgetter(*top25indices.numpy().tolist())(accPerClass_verify)


# In[15]:


accDict = {}
accDict['accPerClass_model1'] = [0.0]*totalClassNum
accPerClass_verify_top25 = [0.0]*totalClassNum
for i in range(A_top25_model1.shape[0]):
    curClsIdx = A_top25_model1[i]
    curClsIdx = int(curClsIdx)
    predClsIdxPred = B_top25_model1[i]
    predClsIdxPred = int(predClsIdxPred)
    accDict['accPerClass_model1'][curClsIdx] += float(curClsIdx==predClsIdxPred)
    accPerClass_verify_top25[curClsIdx] += 1

for i in range(totalClassNum):
    accDict['accPerClass_model1'][i] = accDict['accPerClass_model1'][i] * 1.0 / (accPerClass_verify_top25[i]+0.00001)
    

    
    

accDict['accPerClass_model2'] = [0.0]*totalClassNum
for i in range(A_top25_model2.shape[0]):
    curClsIdx = A_top25_model2[i]
    curClsIdx = int(curClsIdx)
    predClsIdxPred = B_top25_model2[i]
    predClsIdxPred = int(predClsIdxPred)
    accDict['accPerClass_model2'][curClsIdx] += float(curClsIdx==predClsIdxPred)

for i in range(totalClassNum):
    accDict['accPerClass_model2'][i] = accDict['accPerClass_model2'][i] * 1.0 / (accPerClass_verify_top25[i]+0.00001)
    


# In[16]:


tmpval = -np.asarray(list(numPerCls))
sort_index = torch.from_numpy(tmpval)
sort_index = sort_index.sort().indices.numpy().tolist()

nameList_sorted = itemgetter(*sort_index)(nameList)
numPerCls_sorted = itemgetter(*sort_index)(numPerCls) 
acc_mode1_sorted = itemgetter(*sort_index)(accDict['accPerClass_model1'])
acc_mode2_sorted = itemgetter(*sort_index)(accDict['accPerClass_model2'])


# In[17]:


x = np.arange(len(nameList_sorted))  # the label locations
width = 0.4  # the width of the bars


fig, ax = plt.subplots(1, sharex=True, figsize=(18,5), dpi=320)
rects1 = ax.bar(x - width/2, acc_mode1_sorted, width, label='w/o reweighting')
#rects2 = ax.bar(x + width/2, acc_mode2_sorted, width, label='w/ reweighting')
rects3 = ax.bar(x + width/2, acc_mode2_sorted, width, label='w/ reweighting')


ax.set_ylabel('accuracy', size=25)
ax.set_title('accuracy sorted by number of samples', size=25)
ax.set_xticks(x)
ax.set_xticklabels(nameList_sorted, size=15)
ax.legend(fontsize=17)
plt.xticks(rotation=45, ha='right')

ax.tick_params(axis='both', which='major', labelsize=17)


fig.tight_layout()
plt.show()


# In[19]:


x = np.arange(len(numPerCls_sorted))  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots(1, sharex=True, figsize=(18,5), dpi=320)
rects = ax.bar(x, numPerCls_sorted, width, label='w/o reweighting')

ax.set_ylabel('#samples', size=25)
ax.set_title('#samples per class', size=25)
ax.set_xticks(x)
ax.set_xticklabels(nameList_sorted, size=15)
ax.legend(fontsize=17)
plt.xticks(rotation=45, ha='right')

ax.tick_params(axis='both', which='major', labelsize=17)


fig.tight_layout()
plt.show()


# bar chart
# -----
# 
# https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

# 
# 
# Leaving Blank
# -----

# In[ ]:




