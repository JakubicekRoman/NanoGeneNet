# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:05:44 2021

@author: jakubicek
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 16:30:52 2021

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
import h5py

import loaders
    

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels = in_ch,  out_channels = out_ch , kernel_size = 3, stride = 1, padding=1, padding_mode='replicate')
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels = out_ch,  out_channels = out_ch , kernel_size = 3, stride = 1, padding=1, padding_mode='replicate')
        self.relu  = nn.ReLU()
        self.BN = nn.BatchNorm1d(in_ch)
    
    def forward(self, x):

        return self.relu( self.conv2( self.relu( self.conv1( self.BN( x ) ) )))


class Encoder(nn.Module):
    def __init__(self, chs=(1,64,128,256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool1d(3, stride=2, padding=1)
    
    def forward(self, x):

        res = x
        for block in self.enc_blocks:
            x = block(x)
            x +=  res.repeat(1,x.shape[1],1)            
            x = self.pool(x)
            res = self.pool(res)
        return x
    
  
class GenNet(nn.Module):
    def __init__(self, enc_chs=(1,64,128,256), lstm_h_size=256, h_size=1024, num_class=1):
        super().__init__()
        self.lstm_layers = 1
        self.lstm_h_size = lstm_h_size
        
        self.encoder     = Encoder(enc_chs)
        self.lstm        = nn.LSTM(enc_chs[-1], lstm_h_size, batch_first=True,num_layers=self.lstm_layers, bidirectional=False, dropout=0.5)    
        
        self.linear1     = nn.Linear(lstm_h_size, h_size, bias=True)
        self.do          = nn.Dropout(p=0.5)
        self.linear2     = nn.Linear(h_size, num_class, bias=True)    
        self.relu  = nn.ReLU()
        # self.sigm = nn.Sigmoid()

    def forward(self, x):
        
        x = x.permute([0,2,1])
        
        y = self.encoder(x)

        y = y.permute([0,2,1])
        
        _,(self.h,self.c)=self.lstm( y , (self.h,self.c) )
        y = self.h
        y = torch.squeeze(y)
        y=self.linear1(y)
        y=self.do(y)
        y=self.relu(y)
        y=self.linear2(y)

        return y
    
    def init_hiden(self,batch):
        self.h=torch.zeros((self.lstm_layers, batch, self.lstm_h_size)).cuda()
        self.c=torch.zeros((self.lstm_layers, batch, self.lstm_h_size)).cuda()
    
    
def CreateDataset(path_data, ind):      

    h5_list = glob.glob( os.path.normpath( path_data + "**/*.h5"))  
    sigs_list = []
    lbl_ist = []
    allele_list = []
    
    h5_list = h5_list[int(ind[0]):int(ind[1])]
    
    for file_path in h5_list:
        f = h5py.File(file_path)
        for a in f.__iter__():
            sigs_list.append({'file_path': file_path, 'tname': a})
            allele_list.append(np.asarray(f[a]['allele']))
            lbl_ist.append(np.asarray(dictGen[file_path.split('\\')[-1].split('_')[0]]).astype(np.float32))
                
    return sigs_list, lbl_ist



dictGen = dict(gapA=0 , infB=1 , mdh=2 , pgi=3 , phoE=4 , rpoB=5 , tonB=6)
 
path_data = 'C:\data\jakubicek/all_MLST_genes_new_format1/train'
train_list, train_lbl_list = CreateDataset(path_data, (0,38))

path_data = 'C:\data\jakubicek/all_MLST_genes_new_format1/test'
test_list_o , _ = CreateDataset(path_data, (0,-1))

batch = 64
classes = 2

net = GenNet( enc_chs=(1,64,128,256,512), lstm_h_size=512, h_size=1024, num_class=classes).cuda()

optimizer = optim.Adam(net.parameters(), lr=0.0001,weight_decay=1e-6)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, verbose=True)



# train_list = train_list[0:len(train_list):10]
# train_list = np.random.permutation( train_list )
# train_lbl_list = train_lbl_list[0:end_gen:10]
# test_list = test_list[0:len(train_list):10]

# w1,_= np.histogram(np.array(lbl_list), bins=np.arange(0,7+1,1)-0.5)
# weight = torch.tensor((w1/np.sum(w1)).astype(np.float32) ).cuda()

train_loss = []
train_LOSS = []
train_acc = []
train_ACC = []
test_acc = []
test_ACC = []


for epch in range(0,100):

    train_list = np.random.permutation( train_list )
    net.train()
    net.zero_grad()
    ii=0
    # train_acc=[]
    # Pred = torch.zeros((batch, 2, 1), dtype=torch.float32).cuda()
    # LBL = torch.zeros((batch, 1), dtype=torch.float32).cuda()

    for ite in range(0, len(train_list)-batch, batch):

        net.train(mode=True)
        net.zero_grad()
     
        # Sig, Lbl = loaders.Load_cut_signal_h5(ite, batch, train_list, dictGen)
        Sig, Lbl = loaders.Load_cut_gen_h5(ite, batch, train_list, dictGen)

        net.init_hiden(batch)
        Pred = net(Sig.cuda())
        Pred = F.softmax(Pred, dim=1)

        # w1,_= np.histogram(np.array(train_lbl_list), bins=np.arange(0,7+1,1)-0.5)
        # weight = torch.tensor((w1/np.sum(w1)).astype(np.float32) ).cuda()
        weight =torch.tensor( (0.6, 0.4) ).cuda()       
        loss = nn.CrossEntropyLoss(weight)( Pred.squeeze(),   torch.squeeze(Lbl.cuda().type(torch.long)) )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        net.zero_grad()
        torch.cuda.empty_cache()
        
        acc = (Lbl.detach().cpu().numpy().squeeze() == (Pred.detach().cpu().numpy().squeeze().argmax(1))).astype(np.dtype(float))   
        train_loss.append(loss.detach().cpu().numpy())
        train_acc.append( np.mean(acc) )
        
        # train_ACC.append(np.mean(train_acc))
        
        if ii%(int((len(train_list)/batch/2)))  == 0:
        # if ii%10 == 0:
            test_list = np.random.permutation( test_list_o )[0:len(test_list_o):20]
            for i in range(0, len(test_list)-batch, batch): 
                with torch.no_grad():
                    net.train(mode=False)
                    # net.zero_grad()
                 
                    # Sig, Lbl = loaders.Load_cut_signal_h5(ite, batch, train_list, dictGen) 
                    Sig, Lbl = loaders.Load_cut_gen_h5(i, batch, test_list, dictGen)
            
                    net.init_hiden(batch)
                    Pred = net(Sig.cuda())
                    Pred = F.softmax(Pred, dim=1)
                    net.zero_grad()
                    
                    # w1,_= np.histogram(np.array(lbl_list), bins=np.arange(0,7+1,1)-0.5)
                    # weight = torch.tensor((w1/np.sum(w1)).astype(np.float32) ).cuda()
                    # loss = nn.CrossEntropyLoss()( Pred.squeeze(),   torch.squeeze(Lbl.cuda().type(torch.long)) )
                    
                    acc = (Lbl.detach().cpu().numpy().squeeze() == (Pred.detach().cpu().numpy().squeeze().argmax(1))).astype(np.dtype(float))   
                    test_acc.append( np.mean(acc) )
            train_LOSS.append(np.mean(train_loss))
            plt.figure()
            plt.plot( train_LOSS )
            # plt.plot( train_loss )
            # plt.ylim([0, 1.0])
            plt.show()
            
            train_loss = []
         
            # plt.figure
            # plt.plot( train_loss )
            # # plt.ylim([0, 1.0])
            # plt.show()
            
            # plt.figure
            # plt.plot(train_acc)
            # # plt.ylim([0.0,1])
            # plt.show()
            
            plt.figure()
            plt.plot(Lbl.detach().cpu().numpy())
            plt.plot(Pred[:,1].detach().cpu().numpy())
            plt.show()
            
            train_ACC.append(np.mean(train_acc))
            test_ACC.append(np.mean(test_acc))
            plt.figure()
            plt.plot(train_ACC)
            plt.plot(test_ACC)
            # plt.ylim([0.0,1])           
            plt.show()
        
            train_acc = [] 
            test_acc = []     
            # t = time.time()
        
        ii=ii+1
          
            
    torch.cuda.empty_cache()   
          
    # train_ACC.append(np.mean(train_acc))
    
    scheduler.step()

