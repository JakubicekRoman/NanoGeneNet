# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 12:11:58 2021

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
# import pandas as pd
from torch.nn.utils.rnn import pad_sequence

import utilities
import h5py
import loaders2
    

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
        y = torch.cat((y, C),2)
        
        y=self.linear1(y)
        y=F.relu(y) 
        y=self.do(y)
        y=self.linear2(y)  
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
 

path_data = 'C:\data\jakubicek/all_MLST_genes_new_format1/train'
tl, tll = CreateDataset(path_data, (0,145))
train_list = SelectRandomData(tl,tll,2000)     

path_data = 'C:\data\jakubicek\signals_without_all_mlst_genes'
empty_list, _  = CreateDataset(path_data, (0,54))
empty_list = np.random.permutation( empty_list ).tolist()

for l in range(0,2000):
    train_list.append( empty_list[l] )

path_data = 'C:\data\jakubicek/all_MLST_genes_new_format1/test'
test_list_o , _ = CreateDataset(path_data, (0,-1))
# test_list_o = random.sample(test_list_o , 1000)

for l in range(7000,9000):
    test_list_o.append( empty_list[l] )

# # LSTM training○

# net = NetGEN(enc_chs=(1,16,32,64,128), lstm_h_size=256, h_size=512).cuda()
net = torch.load(r"D:\jakubicek\Bioinformatika\Models\net_v3_6.pt")

# optimizer = optim.Adam(net.parameters(), lr=0.00001, weight_decay=0.9)
optimizer = optim.SGD(net.parameters(), lr=0.00001, weight_decay=0., momentum= 0.95)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1, verbose=True)


train_loss = []
train_LOSS = []
train_dice = []
train_DICE = []
test_dice = [] 
test_DICE = []

batchTrain = 1
batch = batchTrain

for epch in range(0,10):
    net.train()
    ii=0
    # train_list = SelectRandomData(tl,tll,2000)
    train_list = np.random.permutation( train_list )
    # train_list = train_list[0:10000]
    
    for ite in range(0, len(train_list)-batch, batch):
    # for ite in range(0, 10, batch):
        batch = batchTrain
                
        # for bt in range(0,batch):
        net.train()
        # net.zero_grad()
        
        # sample,lbl,_= loaders2.Load_cut_signal_h5(ite, batch, train_list, dictGen)
        sample, lbl, clbl = loaders2.Load_whole_signal_h5(train_list[ite], dictGen)

        net.init_hiden(1)
        
        pred = net(sample.cuda())
        pred = F.softmax(pred, dim=2)  
        # net.zero_grad()

        lbl = lbl.permute([0,2,1]).cuda()
        lbl = F.interpolate(lbl, ( pred.shape[1]))
        # lbl = lbl[:,0,:]
        lbl = lbl.permute([0,2,1])
        
        
        pred = pred.permute([0,2,1])
        lbl = lbl[:,:,0]
        # lbl = lbl.squeeze()
        # weight = torch.tensor((0.05, 0.95)).cuda()
        w1 =  (torch.sum(lbl[:])+0.01) / (lbl.shape[1]*lbl.shape[0] +0.01) 
        # # # if w1<=0.01:
        # # #     w1=torch.tensor(0.7)
        # # # w1 = 0.2
        weight = torch.tensor((w1, 1-w1)).cuda()       
        # weight = torch.tensor((0.01, 0.99)).cuda()  
        
        # loss = nn.CrossEntropyLoss(weight)( pred,  lbl.type(torch.long) )
        
        loss = utilities.dice_loss_torch( pred[0,1,:], lbl[0,:]  )
        
        # loss = utilities.dice_loss_torch( pred[0,:,1],  lbl[0,:,0].type(torch.long)  )
        # loss = loss + ( nn.CrossEntropyLoss(weight)( pred,  lbl.type(torch.long) ) )/batch
    
        # loss = nn.BCEWithLogitsLoss(1-w1)( pred,  lbl ) 
        # torch.cuda.empty_cache()
            
        # GT=np.zeros(pred.shape)
        # GT = lbl.detach().cpu().numpy()
        # GT[:,0,:] = ( 1-lbl.detach().cpu().numpy() )
        # P = pred[:,1,:].detach().cpu().numpy()>0.5
         
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        
        train_dice.append(  utilities.dice_torch(( pred[:,1,:]>0.5 ), lbl ).detach().cpu().numpy()  )
        # train_acc.append( np.mean( np.sum( np.sum( GT==P,2),1) / (GT.shape[1]*GT.shape[2]) )  )        
        train_loss.append( loss.detach().cpu().numpy() )

        torch.cuda.empty_cache()
        
        if ii%(int((len(train_list)/batch)/100))  == 0:
        # if ii%(10)  == 0:
            # train_ACC.append(np.mean(train_acc))
            
            plt.figure
            plt.plot(lbl.detach().cpu().numpy()[0,:])
            plt.plot(pred.detach().cpu().numpy()[0,1,:])
            # plt.plot(P[0,:])
            plt.ylim([0.0,1])
            plt.show() 
            
            # class_acc=np.array((0,))
            class_lbl=np.array((0,))
            class_dice=np.array((0,))
            
            test_list = np.random.permutation( test_list_o )[0:len( test_list_o ):50]  
            batch = 1
            for i in range(0, len(test_list)-batch, batch):
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
                    class_lbl = np.concatenate( (class_lbl, clbl.numpy() ) )
                    class_dice = np.concatenate( (class_dice, dice.numpy() ) )
                    
            torch.cuda.empty_cache()
            
            # hi = utilities.comp_class_acc(class_lbl, class_acc)
            hd = utilities.comp_class_acc(class_lbl, class_dice)
            
            train_LOSS.append( (np.mean(train_loss)) )
            
            plt.figure
            # plt.plot(train_loss)
            plt.plot(train_LOSS)
            # plt.ylim([0, 1.0])
            plt.show()        
            
            test_DICE.append( (np.mean(test_dice)) )
            train_DICE.append( (np.mean(train_dice)) )
            plt.figure
            plt.plot( train_DICE )
            plt.plot( test_DICE )
            # plt.ylim([0.8, 1.0])
            plt.show() 
            
            # plt.figure
            # plt.plot(lbl.detach().cpu().numpy()[0,:])
            # plt.plot(pred.detach().cpu().numpy()[0,0,:])
            # # plt.plot(P[0,:])
            # plt.ylim([0.0,1])
            # plt.show() 
            
            plt.figure
            plt.bar(np.arange(0,8), hd)
            # plt.ylim([0.5, 1.0])
            plt.show() 
            
            # width = 0.35
            # fig, ax = plt.subplots()
            # rects1 = ax.bar(np.arange(0,8) - width/2, hi, width)
            # rects2 = ax.bar(np.arange(0,8) + width/2, hd, width)
            # plt.show() 
            
            train_dice = []
            test_dice = []
            train_loss = []
            
        ii=ii+1
        print('Epocha: ' + str(epch) + ', Iterace: ' + str(ite/(len(train_list))*100) )
   
      
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
        
    # torch.cuda.empty_cache()



