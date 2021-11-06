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
import pandas as pd

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

path_data = 'C:\data\jakubicek/all_MLST_genes_new_format1/test'
test_list_o , _ = CreateDataset(path_data, (0,-1))
# test_list_o = random.sample(test_list_o , 1000)

for l in range(7000,9000):
    test_list_o.append( empty_list[l] )
    


## LSTM training○

# net = NetGEN(enc_chs=(1,16,32,64,128), lstm_h_size=256, h_size=512).cuda()
net = torch.load(r"D:\jakubicek\Bioinformatika\Models\net_v3_9_1.pt")


train_dice = []
train_DICE = []
test_dice = [] 
test_DICE = []

class_lbl=np.array((0,))
class_dice=np.array((0,))

res_table = pd.DataFrame(data=[], columns=['FileName', 'ID_signal', 'Gene', 'Dice', 'Gene position anot orig', 'Gene position anot sub', 'Gene position predicted orig',  'Gene position predicted sub', 'ID_image'] )
                                           

test_list = test_list_o
test_list = np.random.permutation( test_list_o )[0:len( test_list_o ):60]
# test_list = np.random.permutation( test_list_o )[0:20:1]

batch = 1

hf = h5py.File('pst_signals.h5', 'w')

for i in range(0, len(test_list), batch):
# for ite in range(0, 10, batch):
    with torch.no_grad():
        
        net.train(mode=False)            
        # sample, lbl, clbl = loaders2.Load_cut_signal_h5(i, batch, test_list, dictGen)
        sample, lbl, clbl = loaders2.Load_whole_signal_h5(test_list[i], dictGen)
        
        net.init_hiden(batch)            
        pred = net(sample.cuda())
        pred = F.softmax(pred, dim=2)

        lbl = lbl.permute([0,2,1]).cuda()
        lbl = F.interpolate(lbl, ( pred.shape[1]))
        lbl = lbl[:,0,:]
        pred = pred.permute([0,2,1])
        # lbl = lbl.squeeze()
        
        GT = lbl.detach().cpu().numpy()
        P = pred[:,1,:].detach().cpu().numpy()>0.5
        
        dice = torch.tensor(np.zeros((1), dtype=np.float32))
        dice[0] = utilities.dice_torch(torch.tensor(P), torch.tensor(GT))
        test_dice.append(dice.numpy()[0])
        
        # test_acc.append( np.mean( np.sum( np.sum( GT==P,2),1) / (GT.shape[1]*GT.shape[2]) )  )
        # class_acc = np.concatenate((class_acc, ( np.sum( np.sum( GT==P,2),1) / (GT.shape[1]*GT.shape[2]) ) ) )
        # class_lbl = np.concatenate( (class_lbl, clbl.numpy() ) )
        # class_dice = np.concatenate( (class_dice, dice.numpy() ) )
        
        plt.figure
        plt.plot(lbl.detach().cpu().numpy()[0,:])
        plt.plot(pred.detach().cpu().numpy()[0,1,:])
        plt.ylim([0.0,1])
        num = '0000000' + str(i)
        num = num[-6:]
        plt.savefig('D:\\jakubicek\\Bioinformatika\\Models\\Export_Images\\' + 'Image_' + num + '.png' )
        plt.show()
         
        a = test_list[i]['tname'].split('\\')[-1]
        FileName = test_list[i]['file_path'].split('\\')[-1]
        f = h5py.File(test_list[i]['file_path'],'r')
        loc = np.asarray(f[a]['coord']).astype(np.float32)
        loc.sort() 
        
        ind = [ii for ii, x in enumerate(list(lbl.squeeze().detach().cpu().numpy()>0.5)) if x ]    
        if not ind:
            loc_pred_sub = [0,0]
        else:
            loc_pred_sub = [ind[0], ind[-1]]
        
        res_table.loc[(i,'FileName')] =   FileName 
        res_table.loc[(i,'ID_signal')] =  a 
        res_table.loc[(i,'Gene')] = clbl.detach().cpu().numpy()
        res_table.loc[(i,'Dice')] = dice[0].detach().cpu().numpy()
        res_table.loc[(i,'Gene position anot orig')] = loc
        res_table.loc[(i,'Gene position anot sub')] = (loc/16).astype( np.int64 ).tolist()
        res_table.loc[(i,'Gene position predicted orig')] = ( (np.array( loc_pred_sub)*16).astype( np.int64 ) ).tolist()
        res_table.loc[(i,'Gene position predicted sub')] = loc_pred_sub
        res_table.loc[(i,'ID_image')] = num
          
        torch.cuda.empty_cache()
    
        class_lbl = np.concatenate( (class_lbl, clbl.numpy() ) )
        class_dice = np.concatenate( (class_dice, dice.numpy() ) )
        
        print(str(i))
        
        hf.create_dataset(num, data=pred[:,1,:].detach().cpu().numpy())
        
hf.close()

utilities.save_to_excel(res_table, 'D:\\jakubicek\\Bioinformatika\\Models\\Export_Images' , 'Results' + num + '.xlsx')
    
hd = utilities.comp_class_acc(class_lbl, class_dice)       
plt.figure
plt.bar(np.arange(0,8), hd)
# plt.ylim([0.5, 1.0])
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



