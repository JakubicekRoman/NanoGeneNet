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
import torch
import torch.optim as optim
import glob
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
from torch.utils.data import Dataset
import utilities


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
    

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels = 1, out_channels =  4, kernel_size = 7, stride = 1, padding=3, padding_mode='replicate')
        self.conv2 = nn.Conv1d(in_channels = 4, out_channels =  8, kernel_size = 5, stride = 2, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv1d(in_channels = 8, out_channels =  16, kernel_size = 5, stride = 2, padding=2, padding_mode='replicate')
        self.conv4 = nn.Conv1d(in_channels = 16, out_channels =  32, kernel_size = 3, stride = 2, padding=1, padding_mode='replicate')

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)

        # self.linear1=nn.Linear(h_size+numF+1,int(h_size/4), bias=True)
        # self.linear1=nn.Linear(h_size+1, h_size, bias=True)
        
        self.linear1=nn.Linear(1+4+8+16+32,128, bias=True)
        self.linear2=nn.Linear(128,2, bias=True)
        
        self.BN0 = nn.BatchNorm1d(1)
        self.BN1 = nn.BatchNorm1d(4)
        self.BN2 = nn.BatchNorm1d(8)
        self.BN3 = nn.BatchNorm1d(16)
        
        # self.MP = nn.MaxPool1d(3, stride=2)
        self.UPS =  nn.Upsample((6250))

    def forward(self, x):

        x = x.permute([0,2,1])
        
        x = self.BN0(x)
        
        y = self.conv1(x)
        y = self.BN1(y)
        # y = self.MP(y)
        y1 = F.relu(y)
        
        
        y = self.conv2(y1)
        # y = self.MP(y)
        y = self.BN2(y)
        y2 = F.relu(y)
        
        y = self.conv3(y2)
        # y = self.MP(y)
        y = self.BN3(y)
        y3 = F.relu(y)
        
        y = self.conv4(y3)
        # y = self.MP(y)
        y4 = F.relu(y)
        
        y = torch.cat( (self.UPS(x), self.UPS(y1), self.UPS(y2), self.UPS(y3), self.UPS(y4) ) , 1 )
        
        y = y.transpose(1, 2)
        y = self.linear1(y)
        y = F.relu(y)
        y = self.linear2(y)
          
        return y

   
batch=8
proc=0.8


path_data = os.path.normpath( 'C:\data\jakubicek\GEN_Data_reload')

sigs_list = glob.glob(os.path.normpath( path_data + "\*.npy"))

N =  np.array( np.shape(sigs_list))
sigs_list = np.random.permutation( sigs_list )

train_list = sigs_list[0:int(np.round(N*proc))]
test_list = sigs_list[int(np.round(int(N)*proc))+1:int(N)]


# # LSTM training○

net = CNN().cuda()

# net = torch.load(r"D:\jakubicek\Bioinformatika\netv3_0.pt")
# net = torch.load(r"D:\jakubicek\Bioinformatika\netv2_0.pt")

# optimizer = optim.Adam(net.parameters(), lr=0.0001,weight_decay=1e-9)
optimizer = optim.SGD(net.parameters(), lr=0.001,weight_decay=1e-8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10*2880/batch, gamma=0.1,verbose=False)
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
           
        # net.init_hiden(batch)
        # net.hidden[0].detach_()
        # net.hidden[1].detach_()
        # net.zero_grad()
        
        sample,lbl = utilities.loaderWin(ite, train_list, batch)
        
        pred = net(sample.cuda())
          
    
        pred = F.softmax(pred, dim=2)
        # loss = utilities.dice_loss(pred, lbl.cuda())
        
        # pred = pred.permute([0,2,1])
        # lbl = lbl.permute([0,2,1])

        lbl = lbl[:,::8,:].cuda()
        # lbl = lbl[:,0:pred.shape[1],:].cuda()

        # loss = torch.mean( -torch.log(pred[lbl==1]) )
        # loss = utilities.WCE_loss(pred, lbl)
        # loss = utilities.dice_loss(pred, lbl)
        # loss = nn.CrossEntropyLoss()(pred, lbl[:,:,0].cuda() )
        loss = nn.BCEWithLogitsLoss()(pred[:,:,0], lbl[:,:,0] )
        # loss = nn.BCEWithLogitsLoss()(pred[:,:,1], lbl.cuda()[:,:,1] )

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
        
        if n%50 == 0:
            
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
            plt.plot(lbl.detach().cpu().numpy()[0,:,0])
            plt.plot(pred.detach().cpu().numpy()[0,:,0])
            # plt.plot(P[0,:])
            # plt.ylim([0.7,1])
            plt.show()    

            train_acc = []                 
            
        n=n+1
        
    torch.cuda.empty_cache()



