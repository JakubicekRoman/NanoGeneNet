# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 14:49:04 2021

@author: jakubicek
"""

import os
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import torch.optim as optim
import glob
import torch.nn as nn
import torch.nn.functional as F 
import torch
import random
import utilities

import h5py
import loaders2

from DetNetGen_3_3 import NetGEN
from DetNetGen_3_3 import Block
from DetNetGen_3_3 import Encoder
 


def CreateDataset(path_data, ind):      

    h5_list = glob.glob( os.path.normpath( path_data + "**/*.h5"))  
    sigs_list = []
    lbl_ist = []
    # allele_list = []
    # number = []
    # ii=0
    # i=0
    
    h5_list = h5_list[int(ind[0]):int(ind[1])]
    
    for file_path in h5_list:
        f = h5py.File(file_path)
        
        for a in f.__iter__():
            sigs_list.append({'file_path': file_path, 'tname': a})
            # allele_list.append(np.asarray(f[a]['allele']))
            lbl_ist.append(np.asarray(dictGen[file_path.split('\\')[-1].split('_')[0]]).astype(np.float32))
            # ii=ii+1
        # number.append( ii )
        # i=i+1
        # ii=0
    return sigs_list, lbl_ist
 
def SelectRandomData(tl,tll,num):    
    train_list=[]
    for num_g in range(0,7):
        t = random.sample( [i  for i,x in enumerate(tll) if x==num_g], num) 
        for ii in t:
            train_list.append(tl[ii])
            
    return train_list
        
    
dictGen = dict(gapA=0 , infB=1 , mdh=2 , pgi=3 , phoE=4 , rpoB=5 , tonB=6, run=7)


## LOding data
 
path_data = 'C:\data\jakubicek\signals_without_all_mlst_genes'
empty_list, _  = CreateDataset(path_data, (0,54))
empty_list = np.random.permutation( empty_list ).tolist()

test_list_o = []

path_data = 'C:\data\jakubicek/all_MLST_genes_new_format1/test'
test_list_o , _ = CreateDataset(path_data, (0,-1))


# for l in range(8000,8500):
#     test_list_o.append( empty_list[l] )

## LSTM training○

# net = NetGEN(enc_chs=(1,16,32,64,128), lstm_h_size=256, h_size=512).cuda()
net = torch.load(r"D:\jakubicek\Bioinformatika\Models\net_v3_6.pt")


test_acc = [] 
test_ACC = []
class_acc = []
class_lbl = []

batch = 4
test_list = test_list_o


for i in range(0, len(test_list)-batch, batch):
# for ite in range(0, 10, batch):
    with torch.no_grad():
        
        net.train(mode=False)            
        sample, lbl, clbl = loaders2.Load_cut_signal_h5(i, batch, test_list, dictGen)

        net.init_hiden(batch)            
        pred = net(sample.cuda())
        pred = F.softmax(pred, dim=2)             

        lbl = lbl.permute([0,2,1]).cuda()
        lbl = F.interpolate(lbl, ( pred.shape[1]))
        pred = pred.permute([0,2,1])
        lbl = lbl.squeeze()
        
        GT=np.zeros(pred.shape)
        GT[:,1,:] = lbl.detach().cpu().numpy()
        GT[:,0,:] = ( 1-lbl.detach().cpu().numpy() )
        P = pred.detach().cpu().numpy()>0.5
        
        test_acc.append( np.mean( np.sum( np.sum( GT==P,2),1) / (GT.shape[1]*GT.shape[2]) )  )
        class_acc = np.concatenate((class_acc, ( np.sum( np.sum( GT==P,2),1) / (GT.shape[1]*GT.shape[2]) ) ) )
        class_lbl = np.concatenate( (class_lbl, clbl.numpy() ) )
        
        plt.figure
        plt.plot(lbl.detach().cpu().numpy()[0,:])
        plt.plot(pred.detach().cpu().numpy()[0,1,:])
        # plt.plot(P[0,:])
        plt.ylim([0.0,1])
        plt.show() 
                    
torch.cuda.empty_cache()

hi = utilities.comp_class_acc(class_lbl, class_acc)

plt.figure
plt.bar(np.arange(0,8), hi)
# plt.ylim([0, 1.0])
plt.xlim([0, 8.0])
plt.show() 

plt.figure
plt.bar(np.arange(0,len(test_acc)), test_acc)
# plt.ylim([0, 1.0])
plt.show() 
       
    
    
    
    
    

    # n=0   
    # for m in range(0, len(test_list), 1):
           
    #     net.init_hiden(batch)
    #     net.train(mode=False)
    #     # net.zero_grad()
        
    #     sample,lbl = loaderWin(m, test_list, batch )
        
    #     pred = net(sample.cuda())
    #     net.zero_grad()
          
    #     pred = F.softmax(pred, dim=2) 
    
    #     lbl = lbl.permute([0,2,1]).cuda()
    #     lbl = F.interpolate(lbl, ( pred.shape[1]))
    #     lbl = lbl[:,0,:]
    
    #     pred = pred.permute([0,2,1])
        
    #     # loss = nn.CrossEntropyLoss(weight=torch.tensor((0.1, 0.9)).cuda() )( pred,  lbl.type(torch.long) )
    
    #     GT = lbl.detach().cpu().numpy()
    #     P = pred[:,1,:].detach().cpu().numpy()>0.5
    #     test_acc.append( np.mean( np.sum( GT==P , 1) / GT.shape[1] ) )
    
    #     torch.cuda.empty_cache()
        
    #     if n%100 == 0:
            
    #         plt.figure
    #         plt.plot(test_acc)
    #         plt.ylim([0.0,1])
    #         plt.show()
            
    #         # plt.figure
    #         # plt.plot(lbl.detach().cpu().numpy()[0,:])
    #         # plt.plot(pred.detach().cpu().numpy()[0,1,:])
    #         # # plt.plot(P[0,:])
    #         # # plt.ylim([0.7,1])
    #         # plt.show()        
            
    #     n=n+1
        
    # torch.cuda.empty_cache()



