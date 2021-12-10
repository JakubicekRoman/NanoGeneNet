# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 16:23:47 2021

@author: jakubicek
"""


import torch.nn as nn
import torch.nn.functional as F 
import torch


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels = in_ch,  out_channels = out_ch , kernel_size = 3, stride = 1, padding=1, padding_mode='replicate')
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels = out_ch,  out_channels = out_ch , kernel_size = 3, stride = 1, padding=1, padding_mode='replicate')
        self.relu  = nn.ReLU()
        self.BN = nn.BatchNorm1d(out_ch)
    
    def forward(self, x):

        return self.BN( self.relu( self.conv2( self.relu( self.conv1( x ) ) )))
    

class Encoder(nn.Module):
    def __init__(self, chs=(1,64,128,256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool1d(3, stride=2, padding=1)
    
    def forward(self, x):

        for block in self.enc_blocks:
            x = block(x)
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
        self.relu  = nn.ReLU()

    def forward(self, x):

        x = x.permute([0,2,1])
        # x = F.normalize(x)
        y = self.encoder(x)
        y = y.permute([0,2,1])
        
        y,(self.h,self.c)=self.lstm( y , (self.h,self.c) )
        
        C = self.c.permute([1,0,2]).repeat(1,y.shape[1],1)
        feat = torch.cat((y, C),2)
        
        y=self.linear1(feat)
        y=F.relu(y) 
        y=self.do(y)
        y=self.linear2(y)  
        
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
        self.linear3     = nn.Linear(h_size, 8, bias=True)
        
        self.relu  = nn.ReLU()

    def forward(self, x ):

        x = x.permute([0,2,1])
        y = self.encoder(x)
        y = y.permute([0,2,1])
        
        _,(self.h,self.c)=self.lstm( y , (self.h,self.c) )
        
        y = torch.squeeze( self.h )
        y=self.linear1(y)
        y=F.relu(y) 
        y=self.do(y)
        y=self.linear3(y)  
        
        return y
    
    
    def init_hiden(self,batch):
        self.h=torch.zeros((self.lstm_layers, batch, self.lstm_h_size)).cuda()
        self.c=torch.zeros((self.lstm_layers, batch, self.lstm_h_size)).cuda()             