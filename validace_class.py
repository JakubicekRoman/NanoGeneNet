# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:44:42 2021

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
import pickle
import time

import h5py
import loaders2

from DetNetGen_3_3 import NetGEN
from DetNetGen_3_3 import Block
from DetNetGen_3_3 import Encoder
 

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels = in_ch,  out_channels = out_ch , kernel_size = 3, stride = 1, padding=1, padding_mode='replicate')
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels = out_ch,  out_channels = out_ch , kernel_size = 3, stride = 1, padding=1, padding_mode='replicate')
        self.relu  = nn.ReLU()
        self.BN = nn.BatchNorm1d(out_ch)
    
    def forward(self, x):

        # return self.relu( self.conv2( self.relu( self.conv1( self.BN( x ) ) )))
        return self.BN( self.relu( self.conv2( self.relu( self.conv1( x ) ) )))
    

class Encoder(nn.Module):
    def __init__(self, chs=(1,64,128,256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool1d(3, stride=2, padding=1)
    
    def forward(self, x):

        # res = x
        for block in self.enc_blocks:
            x = block(x)
            # x +=  res.repeat(1,x.shape[1],1)            
            # res = self.pool(res)
            x = self.pool(x)
        return x
    

class NetGEN(nn.Module):
    def __init__(self, enc_chs=(1,64,128,256), lstm_h_size=256, h_size=1024):
        super(NetGEN, self).__init__()
        self.lstm_layers = 1
        self.h_size = h_size
        self.lstm_h_size = lstm_h_size        
        self.encoder     = Encoder(enc_chs)
        self.lstm        = nn.LSTM(enc_chs[-1], lstm_h_size, batch_first=True, num_layers=self.lstm_layers, bidirectional=False, dropout=0.5)            
        self.linear1     = nn.Linear(lstm_h_size*2, h_size)
        self.do          = nn.Dropout(p=0.5)
        self.linear2     = nn.Linear(h_size, 2, bias=True)
        # self.linear3     = nn.Linear(h_size, 1, bias=True)
        self.relu  = nn.ReLU()

    def forward(self, x):

        x = x.permute([0,2,1])
        # x = F.normalize(x)
        y = self.encoder(x)
        y = y.permute([0,2,1])
        
        y,(self.h,self.c)=self.lstm( y , (self.h,self.c) )
        
        # y = torch.squeeze(y)
        # y=self.linear1(torch.cat((x,y,yC),2))   ### concatenation of input and lstm output  - "residual conection"\
        # y=self.linear1(torch.cat((x,y),2))   ### concatenation of ☺ and lstm output  - "residual conection"\
        C = self.c.permute([1,0,2]).repeat(1,y.shape[1],1)
        feat = torch.cat((y, C),2)
        
        y=self.linear1(feat)
        y=F.relu(y) 
        y=self.do(y)
        y=self.linear2(y)  
        # y=nn.Sigmoid(y)
        # y=F.relu(y) 
        
        return y, feat
    
    
    def init_hiden(self,batch):
        self.h=torch.zeros((self.lstm_layers, batch, self.lstm_h_size)).cuda()
        self.c=torch.zeros((self.lstm_layers, batch, self.lstm_h_size)).cuda()         


        
class ClassGEN(nn.Module):
    def __init__(self, enc_chs=(1,64,128,256), lstm_h_size=256, h_size=1024):
        super(ClassGEN, self).__init__()
        self.lstm_layers = 1
        self.h_size = h_size
        self.lstm_h_size = lstm_h_size      
        
        self.encoder     = Encoder(enc_chs)
        self.lstm        = nn.LSTM(enc_chs[-1], lstm_h_size, batch_first=True, num_layers=self.lstm_layers, bidirectional=False, dropout=0.5)            
        self.linear1     = nn.Linear(lstm_h_size, h_size)
        self.do          = nn.Dropout(p=0.5)
        # self.linear2     = nn.Linear(h_size, h_size, bias=True)
        self.linear3     = nn.Linear(h_size, 8, bias=True)
        
        self.relu  = nn.ReLU()

    def forward(self, x ):

        x = x.permute([0,2,1])
        # x = F.normalize(x)
        y = self.encoder(x)
        y = y.permute([0,2,1])
        
        _,(self.h,self.c)=self.lstm( y , (self.h,self.c) )
        
        # y = torch.squeeze(y)
        # y=self.linear1(torch.cat((x,y,yC),2))   ### concatenation of input and lstm output  - "residual conection"\
        # y=self.linear1(torch.cat((x,y),2))   ### concatenation of ☺ and lstm output  - "residual conection"\
        # C = self.c.permute([1,0,2]).repeat(1,y.shape[1],1)
        
        # y = torch.cat((self.h, pred),2)
        y = torch.squeeze( self.h )
        y=self.linear1(y)
        # y=F.relu(y) 
        # y=self.linear2(y)
        y=F.relu(y) 
        y=self.do(y)
        y=self.linear3(y)  
        # y=nn.Sigmoid(y)
        # y=F.relu(y) 
        
        return y
    
    
    def init_hiden(self,batch):
        self.h=torch.zeros((self.lstm_layers, batch, self.lstm_h_size)).cuda()
        self.c=torch.zeros((self.lstm_layers, batch, self.lstm_h_size)).cuda()  


def CreateDataset(path_data, ind):      

    h5_list = glob.glob( os.path.normpath( path_data + "**/*.h5"))  
    sigs_list = []
    lbl_ist = []
    # allele_list = []
    # number = []
    # ii=0
    # i=0
    
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
# empty_list = np.random.permutation( empty_list ).tolist()

path_data = 'C:\data\jakubicek/all_MLST_genes_new_format1/test'
test_list_o , _ = CreateDataset(path_data, (0,-1))
# test_list_o = random.sample(test_list_o , 1000)

for l in range(7000,9000):
    test_list_o.append( empty_list[l] )
    

path_save = 'D:\\jakubicek\\Bioinformatika\\Models\\Export_Images_Class_5_2_1'

## LSTM training○

net1 = torch.load(r"D:\jakubicek\Bioinformatika\Models\net_v3_9_1.pt")
net2 = torch.load(r"D:\jakubicek\Bioinformatika\Models\net_v5_2_1.pt")


test_acc = [] 
test_clbl = []

class_lbl=np.array((0,))
class_acc=np.array((0,))

res_table = pd.DataFrame(data=[], columns=['FileName', 'ID_signal', 'Gene', 'Predicted Gene', 'Predicted probalities' ] )                                         

test_list = test_list_o
# test_list = np.random.permutation( test_list_o )[0:len( test_list_o ):1]
# test_list = np.random.permutation( test_list_o )[0:20:1]

batch = 1

T = []
N = []

# hf = h5py.File('D:\jakubicek\Bioinformatika\Models\Export_Images\pst_signals.h5', 'w')
# hf = h5py.File('D:\jakubicek\Bioinformatika\Models\Export_Images\info.h5', 'w')

for i in range(0, len(test_list), batch):
# for ite in range(0, 10, batch):
    with torch.no_grad():
    
        net1.train(mode=False) 
        net2.train(mode=False) 
        # sample, lbl, clbl = loaders2.Load_cut_signal_h5(i, batch, test_list, dictGen)
        sample, lbl, clbl = loaders2.Load_whole_signal_h5(test_list[i], dictGen)
        
        N.append(sample.shape[1])
        
        t = time.time()
        
        net1.init_hiden(batch)            
        pred, feat = net1(sample.cuda())
        pred = F.softmax(pred, dim=2)
        
        feat = torch.cat((feat[:,:,0:256], torch.unsqueeze( pred[:,:,1],2 ) ),2)
        
        net2.init_hiden(batch)
        pred2 = net2(feat.cuda())
        pred2 = F.softmax(pred2, dim=0)
        
        T.append(time.time() - t)

        acc = (clbl.detach().cpu().numpy().squeeze() == ( pred2.detach().cpu().numpy().squeeze().argmax(0) )).astype(np.dtype(float))   
        test_acc.append( acc )
        test_clbl.append(  clbl.numpy() ) 
         
        a = test_list[i]['tname'].split('\\')[-1]
        FileName = test_list[i]['file_path'].split('\\')[-1]
        # f = h5py.File(test_list[i]['file_path'],'r')
   
        
        res_table.loc[(i,'FileName')] =   FileName 
        res_table.loc[(i,'ID_signal')] =  a 
        res_table.loc[(i,'Gene')] = int(clbl.detach().cpu().numpy())
        res_table.loc[(i,'Predicted Gene')] = pred2.detach().cpu().numpy().squeeze().argmax(0)   
        res_table.loc[(i,'Predicted probalities')] = pred2.detach().cpu().numpy().squeeze() 
        # res_table.loc[(i,'Gene position anot orig')] = loc
        
          
        torch.cuda.empty_cache()
    
        class_lbl = np.concatenate( (class_lbl, clbl.numpy() ) )
        class_acc = np.concatenate( (class_acc, [np.array( acc )] ) )
        
        print(str(i/len(test_list)*100))
        
        # hf.create_dataset(num, data=pred[:,1,:].detach().cpu().numpy())
        
# hf.close()

utilities.save_to_excel(res_table, path_save, 'Results.xlsx')
   

hd = utilities.comp_class_acc(class_lbl, class_acc)       
plt.figure
plt.bar(np.arange(0,8), hd)
# plt.ylim([0.5, 1.0])  
plt.savefig( path_save + '\\GenesBar' + '.png' )
plt.show() 
plt.close() 

print( np.mean(class_acc) )
    

ind = np.argsort(N)
N1 = np.array(N)[ind]
T1 = np.array(T)[ind]
plt.figure()
plt.plot(N1,T1)  
  

res_table.to_pickle(path_save + '\\res_table.pickle')

open_file = open(path_save + '\\test_list.pickle', "wb")
pickle.dump(test_list, open_file)
open_file.close()

hf = h5py.File(path_save + '\info.h5', 'w')
hf.create_dataset('times' , data=T  )
hf.create_dataset('lenghts' , data=N  )  
hf.create_dataset('Acc_genes' , data=hd  )  
hf.close()

    
## # time demanding 
# hf = h5py.File('D:\jakubicek\Bioinformatika\Models\Export_Images\info.h5', 'r')
# N = np.asarray(hf['lenghts']).astype(np.float32)
# times = np.asarray(hf['times']).astype(np.float32)

# ind = np.argsort(N)

# N = N[ind]
# times = times[ind]

# plt.figure()
# plt.plot(N,times)







