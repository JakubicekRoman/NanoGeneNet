
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 16:30:52 2021

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
# from torch.utils.data import DataLoader 
# from torch.utils.data import Dataset
import utilities
import torch
# from torch import torchaudio 
import torchaudio
import random

import loaders
    


class NetGEN(nn.Module):
    def __init__(self):
        super(NetGEN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels = 1,  out_channels = 16 , kernel_size = 3, stride = 1, padding=1, padding_mode='replicate')
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
          
        # return y

   
batch=8
proc=0.95


path_data = os.path.normpath( 'C:\data\jakubicek\GEN_Data_reload')

sigs_list = glob.glob(os.path.normpath( path_data + "\*.npy"))

N =  np.array( np.shape(sigs_list))
sigs_list = np.random.permutation( sigs_list )

train_list = sigs_list[0:int(np.round(N*proc))]
test_list = sigs_list[int(np.round(int(N)*proc))+1:int(N)]


# # LSTM training○

net = NetGEN().cuda()
# net = torch.load(r"D:\jakubicek\Bioinformatika\netv5_0.pt")

# net = torch.load(r"D:\jakubicek\Bioinformatika\netv3_0.pt")
# net = torch.load(r"D:\jakubicek\Bioinformatika\netv2_0.pt")

optimizer = optim.Adam(net.parameters(), lr=0.00001, weight_decay=1e-6)
# optimizer = optim.SGD(net.parameters(), lr=0.0001, weight_decay=1e-6)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1*3593/batch, gamma=0.1, verbose=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1, verbose=False)
# net.init_hiden(batch)

train_loss = []
train_acc = []
train_ACC = []

for epch in range(0,20):
    net.train()
    ii=0
    iii=1
    indx=0
    train_list = np.random.permutation( train_list )
    
    # net.init_hiden(batch)
    
    for ite in range(0, len(train_list), 1):
    # for ite in range(0, 10, batch):
        net.train()
        net.zero_grad()
        batch=8
        iii=1
        if iii==1:
            sample,lbl = loaders.loaderWinGen(indx, train_list, batch, mode='interp')
            indx = indx+batch
        elif iii==2:   
            sample,lbl = loaders.loaderWinRand(indx, train_list, batch, mode='interp')
            indx = indx+batch
        elif iii==3:
            sample,lbl = loaders.loaderWhole(indx, train_list, 1)
            iii=0
            indx = indx+1
            batch=1

        net.init_hiden(batch)
        
        pred = net(sample.cuda())
        pred = F.softmax(pred, dim=2)             

        lbl = lbl.permute([0,2,1]).cuda()
        lbl = F.interpolate(lbl, ( pred.shape[1]))
        lbl = lbl[:,0,:]
        # lbl = lbl.permute([0,2,1])
        pred = pred.permute([0,2,1])
        
        # weight = torch.tensor((0.05, 0.95)).cuda()
        w1 =  (torch.sum(lbl[0,:])+0.0001) / (lbl.shape[1] +0.0001) 
        weight = torch.tensor((w1, 1-w1)).cuda()
        loss = nn.CrossEntropyLoss(weight)( pred,  lbl.type(torch.long) ) 

        GT = lbl.detach().cpu().numpy()
        P = pred[:,1,:].detach().cpu().numpy()>0.5
        train_acc.append( np.mean( np.sum( GT==P , 1) / GT.shape[1] )  )
        
        train_loss.append(loss.detach().cpu().numpy())
     
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        torch.cuda.empty_cache()
        
        if ii%(int((len(train_list)/batch)/10))  == 0:
            
            train_ACC.append(np.mean(train_acc))
            
            # plt.figure
            # plt.plot(train_loss)
            # plt.ylim([0, 1.0])
            # plt.show()
            
            plt.figure
            plt.plot(-np.log(train_ACC))
            # plt.ylim([0.0,1])
            plt.show()
            
            plt.figure
            plt.plot(lbl.detach().cpu().numpy()[0,:])
            plt.plot(pred.detach().cpu().numpy()[0,1,:])
            # plt.plot(P[0,:])
            plt.ylim([0.0,1])
            plt.show()    

            train_acc = []                 
            
        ii=ii+1
        iii=iii+1
        
        if indx+8 > len(train_list):
            break
        
    scheduler.step()

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
        
    torch.cuda.empty_cache()



