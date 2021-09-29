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
import torch.optim as optim
import glob
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
from torch.utils.data import Dataset
# import utilities


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
        
        sig = sig[::4,:]
        lbl = lbl[::4,:]
        
        return  sig, lbl
    


class LSTM(nn.Module):
    def __init__(self,x_size,h_size,y_size,lstm_layers=2,dropout=0.5):
        super(LSTM, self).__init__()

        self.conv1 = nn.Conv1d(in_channels = 1, out_channels =  8, kernel_size = 3, stride = 1, padding=1, padding_mode='replicate')
        self.lstm_layers=lstm_layers
        self.h_size=h_size

        self.lstm=nn.LSTM(x_size,h_size,batch_first=True,num_layers=self.lstm_layers,dropout=dropout)    

        # self.linear1=nn.Linear(h_size+x_size,h_size)
        self.linear1=nn.Linear(h_size,h_size)
        
        self.do=nn.Dropout(p=dropout)
        # self.linear2=nn.Linear(h_size,h_size)
        self.linear3=nn.Linear(h_size,y_size)
        

    def forward(self, x):

        x = x.permute([0,2,1])
        y=self.conv1(x)
        y = F.relu(y)
        y = y.permute([0,2,1])
        x = x.permute([0,2,1])
        
        y,(self.h,self.c)=self.lstm(y,(self.h,self.c))

        # y=self.linear1(torch.cat((x,y),2))   ### concatenation of input and lstm output  - "residual conection"
        y =self.linear1(y)
        y=F.relu(y)
        y=self.do(y)

        # y=self.linear2(y)
        # y=F.relu(y)

        y=self.linear3(y)
            
        return y

    def init_hiden(self,batch):
        self.h=torch.zeros((self.lstm_layers, batch, self.h_size)).cuda()
        self.c=torch.zeros((self.lstm_layers, batch, self.h_size)).cuda()


   
batch=1
hiden_dim=100
proc = 0.7

path_data = os.path.normpath( 'C:\Data\Jakubicek\Bioinformatika\Data_reload')

N =  np.shape( glob.glob(os.path.normpath( path_data + "\*.npy")))[0]
ind = np.random.permutation(np.arange(0,N))

dataset = nacitac(path_data, ind[0:int(np.round(N*proc))]  )
train_loader = DataLoader(dataset, shuffle=True, batch_size=batch)

dataset = nacitac(path_data,  ind[int(np.round(N*proc))+1:N]  )   
test_loader = DataLoader(dataset, shuffle=True, batch_size=batch)


# # LSTM training

net = LSTM(8, hiden_dim, 2).cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001,weight_decay=1e-6)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.8,verbose=False)
net.init_hiden(batch)


train_loss = []
test_loss = []
train_acc = []
test_acc = []


net.train()

for i,(sample, lbl) in enumerate(train_loader):
       
    net.init_hiden(batch)
    
    pred = net(sample.cuda())
    pred = F.softmax(pred, dim=2)
    # loss = utilities.dice_loss(pred, lbl.cuda())
    
    loss = torch.mean( -torch.log(pred[lbl==1]) )
    train_loss.append(loss.detach().cpu().numpy()) 
 
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1)
    optimizer.step()
    scheduler.step()
    
    torch.cuda.empty_cache()
    
    plt.figure
    plt.plot(train_loss)
    plt.show()
    
    plt.figure
    plt.plot(lbl.detach().cpu().numpy()[0,:,0])
    plt.plot(pred.detach().cpu().numpy()[0,:,0])
    plt.show()

torch.cuda.empty_cache()


# plt.figure
# plt.plot(lbl[:,0])
# plt.plot(pred[:,0])

