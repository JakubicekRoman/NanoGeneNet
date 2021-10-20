# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 18:36:07 2021

@author: jakubicek
"""

import os
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
# import pandas as pd

import torch.optim as optim
import glob
import torch.nn as nn
import torch.nn.functional as F 
import torch
import random
import h5py


def loader(ite, sig_list, batch):
 
    n  = 50000
    LBL = torch.tensor(np.zeros((batch,int(n),2), dtype=np.float32))
    Sig = torch.tensor(np.zeros((batch,int(n),1), dtype=np.float32))
    
    for i in range(0,batch):
        # sig, loc = np.load( sig_list[ite+i], allow_pickle=True )
        
        f = h5py.File( sig_list[ite+i], 'r')
        a = list(f.keys())[0]
        sig = np.asarray(f[a]['signal'])
        loc = list(f[a]['coord'])
            
        # for a in f.__iter__():
            # sig = np.asarray(f[a]['signal'])Y
            # loc = list(f[a]['coord'])

            
        
        N = len(sig)
        loc.sort()
        lbl = np.zeros([N,1], dtype=bool)
        lbl[loc[0]:loc[1],0] = True
        lbl =  np.float32(lbl)
        # sig = np.expand_dims(sig,1)
        
        if N < n:
            sig_ = np.interp(np.linspace(0,N-1,n), np.linspace(0,N-1,N), sig)  
            lbl_ = np.interp(np.linspace(0,N-1,n), np.linspace(0,N-1,N), lbl[:,0])
        else:      
            z = loc[1]-n
            if z<0:
                z=0
            k=loc[0]-1
            if k+n  >sig.shape[0]:
                k=sig.shape[0]-n-1               

            M = random.randrange(z, k)
            sig_ = sig[range(int(M),int(M)+n)]
            lbl_ = lbl[range(int(M),int(M)+n)]
            
        lbl = np.zeros([n,2]).astype(np.float32)
        lbl[:,0] = lbl_.T
        lbl[:,1] = (~lbl_[:].astype(np.bool_)).astype(np.float32).T
        sig = np.expand_dims(sig_ ,1)
        # plt.plot(lbl)
        # plt.show()
            
        Sig[i,:,:] = torch.tensor(sig.astype(np.float32))
        LBL[i,:,:] = torch.tensor(lbl.astype(np.float32))
    
    return  Sig, LBL

class NetGEN(nn.Module):
    def __init__(self):
        super(NetGEN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 16 , kernel_size = 3, stride = 1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv1d(in_channels = 16, out_channels = 32 , kernel_size = 3, stride = 1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv1d(in_channels = 32, out_channels = 64 , kernel_size = 3, stride = 1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv1d(in_channels = 64, out_channels = 128 , kernel_size = 3, stride = 1, padding=1, padding_mode='replicate')
        self.lstm_layers = 1
        self.h_size = h_size = 256 
        
        # self.Spect = torchaudio.transforms.Spectrogram(n_fft=(1024), win_length=(100), hop_length=(100) , pad=(0), pad_mode='reflect', normalized=True, power=1 )

        # self.lstm=nn.LSTM(numF+1,h_size,batch_first=True,num_layers=self.lst↑m_layers,dropout=dropout, bidirectional=False)    
        self.lstm=nn.LSTM(128, h_size, batch_first=True,num_layers=self.lstm_layers, bidirectional=False, dropout=0.5)    

        # self.linear1=nn.Linear(h_size+numF+1,int(h_size/4), bias=True)
        # self.linear1=nn.Linear(h_size+1, h_size, bias=True)
        self.linear1=nn.Linear(h_size*2,h_size*2)
        
        self.do=nn.Dropout(p=0.5)
        self.MP = nn.MaxPool1d(3, stride=2, padding=1)
        # self.linear2=nn.Linear(int(h_size/4),h_size, bias=True)
        self.linear3=nn.Linear(h_size*2, 2, bias=True)
        
        self.BN1 = nn.BatchNorm1d(16)
        self.BN2 = nn.BatchNorm1d(32)
        self.BN3 = nn.BatchNorm1d(64)
        self.BN4 = nn.BatchNorm1d(128)
        
        # self.UPS =  nn.Upsample((1/4))    
    def forward(self, x):
    
        x = x.permute([0,2,1])
        
        y = self.conv1(x)
        y = F.relu(y)
        y = self.BN1(y)
        y1 = self.MP(y)
        
        y = self.conv2(y1)
        y = F.relu(y)
        y = self.BN2(y)
        y2 = self.MP(y)
        
        y = self.conv3(y2)
        y = F.relu(y)
        y = self.BN3(y)
        y3 = self.MP(y)
        
        y = self.conv4(y3)
        y = F.relu(y)
        y = self.BN4(y)
        y4 = self.MP(y)
        
        # self.UPS =  nn.Upsample(  [ *list( y.size())[0:2]  ,  *[list(y4.size())[2]]  ]   )
        
        # y = torch.cat( ( self.UPS(x) , self.UPS(y1), self.UPS(y2), self.UPS(y3), self.UPS(y4) ) , 1 )
            
        y = y4.permute([0,2,1])
              
        y,(self.h,self.c)=self.lstm( y ,(self.h,self.c))
        
        # y,(self.h,self.c)=self.lstm(y,(self.h,self.c))

        # y=self.linear1(torch.cat((x,y,yC),2))   ### concatenation of input and lstm output  - "residual conection"\
        # y=self.linear1(torch.cat((x,y),2))   ### concatenation of ☺ and lstm output  - "residual conection"\
        C = self.c.permute([1,0,2]).repeat(1,y.shape[1],1)
        
        y=self.linear1( torch.cat((y, C),2) )
        y=F.relu(y)
                
        # y =self.linear1(y)
        # y=F.relu(y)
        y=self.do(y)

        # y=self.linear2(y)
        # y=F.relu(y)

        y=self.linear3(y)
        
        # y=F.sigmoid(y)
        
        return y
        
        
    def init_hiden(self,batch):
        self.h=torch.zeros((self.lstm_layers, batch, self.h_size)).cuda()
        self.c=torch.zeros((self.lstm_layers, batch, self.h_size)).cuda()
          
        # return y☺


path_data = os.path.normpath( 'C:\data\jakubicek\gapA_new_format')
sigs_list = glob.glob(os.path.normpath( path_data + "\**\*.h5"), recursive=True)

# N =  np.array( np.shape(sigs_list))
# sigs_list = np.random.permutation( sigs_list )

net = torch.load(r"D:\jakubicek\Bioinformatika\netv5_1.pt")


batch = 1
m=16

with torch.no_grad():

    net.init_hiden(batch)
    net.train(mode=False)
    
    sample,lbl = loader(m, sigs_list, batch)
    
    pred = net.eval()(sample.cuda())
    net.zero_grad()
    pred = F.softmax(pred, dim=2)
    
    pred = pred.permute([0,2,1]).cuda()
    pred = F.interpolate(pred, ( lbl.shape[1]))
    
    lbl = lbl[:,:,0]
    
    
    plt.figure
    plt.plot(lbl.detach().cpu().numpy()[0,:])
    plt.plot(pred.detach().cpu().numpy()[0,1,:])
    # plt.plot(P[0,:])
    plt.ylim([0,1])
    plt.show()
    
    torch.cuda.empty_cache()

        
# for m in range(0, len(test_list)-batch, batch):
       
#     net.train(mode=False)
#     # net.zero_grad()
    
#     sample,lbl = loaderWin(m, test_list, batch)
    
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
    
    
#     plt.figure
#     plt.plot(train_acc)
#     plt.ylim([0.0,1])
#     plt.show()
    
#     plt.figure
#     plt.plot(lbl.detach().cpu().numpy()[0,:])
#     plt.plot(pred.detach().cpu().numpy()[0,1,:])
#     # plt.plot(P[0,:])
#     # plt.ylim([0.7,1])
#     plt.show()        
        
    
#     torch.cuda.empty_cache()



