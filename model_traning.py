# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 14:19:18 2021

@author: jakubicek
"""

import os
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
# import pandas as pd
import torch
from torch.utils import data
import torch.optim as optim
import glob
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
from torch.utils.data import Dataset



class nacitac(Dataset):
    def __init__(self, path_data,ind):      
        path_data = os.path.normpath(path_data)
        sigs_list = glob.glob(os.path.normpath( path_data + "\*.npy"))
        
        self.sigs_list = []
        for index in ind:
             self.sigs_list.append(sigs_list[index])
    
    def __len__(self):
        return len(self.sigs_list)
    
    def __getitem__(self, index):        
        sig, loc = np.load( self.sigs_list[index], allow_pickle=True )  
        N = len(sig)
        loc.sort()       
        sig = torch.tensor( sig.astype(np.float32) )
        sig = sig.unsqueeze(1)
        # sig = sig.unsqueeze(0)
        
        lbl = np.zeros([N,2], dtype=bool)
        lbl[loc[0]:loc[1],0] = True
        lbl[:,1] = ~lbl[:,0]
        lbl = torch.tensor(lbl)
        return  sig, lbl
    


class LSTM(nn.Module):
    def __init__(self,x_size,h_size,y_size,lstm_layers=1,dropout=0.5):
        super(LSTM, self).__init__()

        self.lstm_layers=lstm_layers
        self.h_size=h_size

        self.lstm=nn.LSTM(x_size,h_size,batch_first=True,num_layers=self.lstm_layers,dropout=dropout)    

        self.linear1=nn.Linear(h_size+x_size,h_size)#
        self.do=nn.Dropout(p=dropout)
        self.linear2=nn.Linear(h_size,h_size)
        self.linear3=nn.Linear(h_size,y_size)

    def forward(self, x):

        y,(self.h,self.c)=self.lstm(x,(self.h,self.c))

        y=self.linear1(torch.cat((x,y),2))   ### concatenation of input and lstm output  - "residual conection"

        y=F.relu(y)
        y=self.do(y)

        y=self.linear2(y)
        y=F.relu(y)

        y=self.linear3(y)
            
        return y

    def init_hiden(self,batch):
        self.h=torch.zeros((self.lstm_layers, batch, self.h_size)).cuda()
        self.c=torch.zeros((self.lstm_layers, batch, self.h_size)).cuda()

 
def dice_loss(X, Y):
    eps = 1.
    dice = ((2. * torch.sum(X*Y) + eps) / (torch.sum(X) + torch.sum(Y) + eps) )
    return 1 - dice

   
batch=1
hiden_dim=50
proc = 0.7

path_data = os.path.normpath( 'D:\Projekty\Bioinformatika\Data_reload')

N =  np.shape( glob.glob(os.path.normpath( path_data + "\*.npy")))[0]
ind = np.random.permutation(np.arange(0,N))

dataset = nacitac(path_data, ind[0:int(np.round(N*proc))]  )
train_loader = DataLoader(dataset, shuffle=True, batch_size=batch)

dataset = nacitac(path_data,  ind[int(np.round(N*proc))+1:N]  )   
test_loader = DataLoader(dataset, shuffle=True, batch_size=batch)


# # LSTM training

net = LSTM(1, hiden_dim, 2).cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001,weight_decay=1e-6)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.1)
net.init_hiden(batch)


train_loss = []
test_loss = []
train_acc = []
test_acc = []


for i,(sample, lbl) in enumerate(train_loader):
    net.train()
    pred = net(sample.cuda())
    pred = F.softmax(pred, dim=2)
    loss = dice_loss(pred, lbl.cuda())
    train_loss.append(loss.detach().cpu().numpy())

torch.cuda.empty_cache()



# plt.figure
# plt.plot(lbl[:,0])
# plt.plot(pred[:,0])

