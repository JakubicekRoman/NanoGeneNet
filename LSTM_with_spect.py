
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



# class nacitac(Dataset):
#     def __init__(self, path_data,ind):      
#         path_data = os.path.normpath(path_data)
#         sigs_list = glob.glob(os.path.normpath( path_data + "\*.npy"))
        
#         self.sigs_list = []
#         for index in ind:
#              self.sigs_list.append(sigs_list[index])
    
#     def __len__(self):
#         return len(self.sigs_list)
    
#     def __getitem__(self, index):        
#         sig, loc = np.load( self.sigs_list[index], allow_pickle=True )  
#         N = len(sig)
#         loc.sort()       
#         sig = sig.astype(np.float32)
#         # sig = sig.unsqueeze(1)
#         sig = np.expand_dims(sig,1)
#         # sig = sig.unsqueeze(0)
        
#         lbl = np.zeros([N,2], dtype=bool)
#         lbl[loc[0]:loc[1],0] = True
#         lbl[:,1] = ~lbl[:,0]
#         lbl = np.float32(lbl)
        
#         sig = utilities.crop_sig(sig, loc).astype(np.float32)
#         lbl = utilities.crop_sig(lbl, loc).astype(np.float32)
        
#         lbl = torch.tensor(lbl)
#         sig = torch.tensor(sig)
        
#         # sig = sig.unsqueeze(0)
#         # lbl = lbl.unsqueeze(0)
                 
#         # sig = sig[::2,:]
#         # lbl = lbl[::2,:]○
        
#         return  sig, lbl
    

class NetGEN(nn.Module):
    def __init__(self):
        super(NetGEN, self).__init__()

        # self.conv1 = nn.Conv1d(in_channels = 1, out_channels =  numF, kernel_size = 3, stride = 1, padding=1, padding_mode='replicate')
        self.lstm_layers = 1
        self.h_size = h_size = 256 
        
        self.Spect = torchaudio.transforms.Spectrogram(n_fft=(1024), win_length=(100), hop_length=(100) , pad=(0), pad_mode='reflect', normalized=True, power=1 )

        # self.lstm=nn.LSTM(numF+1,h_size,batch_first=True,num_layers=self.lstm_layers,dropout=dropout, bidirectional=False)    
        self.lstm=nn.LSTM(64, h_size, batch_first=True,num_layers=self.lstm_layers, bidirectional=False)    

        # self.linear1=nn.Linear(h_size+numF+1,int(h_size/4), bias=True)
        # self.linear1=nn.Linear(h_size+1, h_size, bias=True)
        self.linear1=nn.Linear(h_size*2,h_size*2)
        
        self.do=nn.Dropout(p=0.5)
        # self.linear2=nn.Linear(int(h_size/4),h_size, bias=True)
        self.linear3=nn.Linear(h_size*2, 2, bias=True)
        
        self.BN = nn.BatchNorm1d(1)


    def forward(self, x):

        x = x.permute([0,2,1])
        x = self.BN(x)
        # x = x.permute([0,2,1])
        
        Sxx = self.Spect(x) 
        Sxx = Sxx[:,0,0:64,:]      
        Sxx = Sxx.permute([0,2,1])
        
        # # plt.imshow( torch.log(Sxx[1,0,0:100,:].detach().cpu()) )
        # plt.figure()
        # plt.imshow( (Sxx[0,0,0:64,:].detach().cpu()) )
        # plt.figure()
        # plt.plot(lbl[0,::100,0].detach().cpu().numpy())
        # plt.show()
        
        y,(self.h,self.c)=self.lstm(Sxx,(self.h,self.c))
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
proc=0.8


path_data = os.path.normpath( 'C:\data\jakubicek\GEN_Data_reload')

sigs_list = glob.glob(os.path.normpath( path_data + "\*.npy"))

N =  np.array( np.shape(sigs_list))
sigs_list = np.random.permutation( sigs_list )

train_list = sigs_list[0:int(np.round(N*proc))]
test_list = sigs_list[int(np.round(int(N)*proc))+1:int(N)]


# # LSTM training○

net = NetGEN().cuda()

# net = torch.load(r"D:\jakubicek\Bioinformatika\netv3_0.pt")
# net = torch.load(r"D:\jakubicek\Bioinformatika\netv2_0.pt")

optimizer = optim.Adam(net.parameters(), lr=0.001,weight_decay=1e-6)
# optimizer = optim.SGD(net.parameters(), lr=0.0001, weight_decay=1e-6)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10*2880/batch, gamma=0.1, verbose=False)
# net.init_hiden(batch)

train_loss = []
test_loss = []
train_acc = []
test_acc = []
train_ACC = []

for epch in range(0,80):
    net.train()
    n=0
    train_list = np.random.permutation( train_list )
    
    # net.init_hiden(batch)
    
    for ite in range(0, len(train_list)-batch, batch):
           
        net.init_hiden(batch)
        # net.hidden[0].detach_()
        # net.hidden[1].detach_()
        # net.zero_grad()
        
        sample,lbl = utilities.loaderWin(ite, train_list, batch)
        
        pred = net(sample.cuda())
          
        pred = F.softmax(pred, dim=2)
        # loss = utilities.dice_loss(pred, lbl.cuda())
        
        # pred = pred.permute([0,2,1])
        # lbl = lbl.permute([0,2,1])

        # lbl = lbl[:,::8,:].cuda()
        # lbl[0,::100,0]
        # lbl = lbl[:,0:pred.shape[1],:].cuda()

        lbl = lbl.permute([0,2,1]).cuda()
        lbl = F.interpolate(lbl, ( pred.shape[1]))
        lbl = lbl[:,0,:]
        # lbl = lbl.permute([0,2,1])
        pred = pred.permute([0,2,1])
        
        
        loss = nn.CrossEntropyLoss(weight=torch.tensor((0.1, 0.9)).cuda() )( pred,  lbl.type(torch.long) )
         
        # loss = torch.mean( -torch.log(pred[lbl==1]) ) + torch.sigmoid(F.mse_loss(pred[:,:,0],lbl[:,:,0]))
        # loss = utilities.WCE_loss(pred, lbl.detach().cpu())
        # loss = utilities.dice_loss(pred, lbl)
        # loss = nn.CrossEntropyLoss()(  pred[:,:,0].long  ,  lbl[:,:,0].long  )
        # loss = nn.BCEWithLogitsLoss()(pred[:,:,0], lbl[:,:,0] )
        # loss = nn.BCEWithLogitsLoss()(pred, lbl ) + torch.sigmoid(F.mse_loss(pred[:,:,0],lbl[:,:,0]))

        # GT = lbl[:,:,0].detach().cpu().numpy()
        # P = pred[:,:,0].detach().cpu().numpy()>0.5
        # train_acc.append( np.mean( np.sum( GT==P , 1) / GT.shape[1] )  )
        
        train_loss.append(loss.detach().cpu().numpy())
     
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        torch.cuda.empty_cache()
        
        if n%20 == 0:
            
            # train_ACC.append(np.mean(train_acc))
            
            plt.figure
            plt.plot(train_loss)
            plt.ylim([0, 1.0])
            plt.show()
            
            # plt.figure
            # plt.plot(train_ACC)
            # plt.ylim([0.7,1])
            # plt.show()
            
            plt.figure
            plt.plot(lbl.detach().cpu().numpy()[0,:])
            plt.plot(pred.detach().cpu().numpy()[0,0,:])
            # plt.plot(P[0,:])
            # plt.ylim([0.7,1])
            plt.show()    

            # train_acc = []                 
            
        n=n+1
        
    torch.cuda.empty_cache()



