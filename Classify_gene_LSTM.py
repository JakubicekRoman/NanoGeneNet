
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
# from pathlib import Path
import torch.optim as optim
import glob
import torch.nn as nn
import torch.nn.functional as F 
# from torch.utils.data import DataLoader 
# from torch.utils.data import Dataset
# import utilities
import torch
# from torch import torchaudio 
# import torchaudio
# import random
import h5py
# import time

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
        # return self.relu( self.conv1( self.BN( x ) ) )
        # y = self.relu( self.conv2( self.relu( self.conv1( self.BN( x ) ) )))
        # y = self.relu( self.BN( self.conv2( self.relu( self.conv1( x ) ) )))
        # y = y + x.repeat(1,y.shape[1],1)
        # y = torch.add(y, x.repeat(1,y.shape[1],1))
        # return torch.add(y, x.repeat(1,y.shape[1],1))
        # return self.relu( self.conv1( x ) )
        return self.relu( self.conv2( self.relu( self.conv1( self.BN( x ) ) )))


class Encoder(nn.Module):
    def __init__(self, chs=(1,64,128,256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool1d(3, stride=2, padding=1)
    
    def forward(self, x):
        # ftrs = []
        res = x
        for block in self.enc_blocks:
            x = block(x)
            
            # ftrs.append(x)
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
 
        # y = enc_ftrs.permute([0,2,1])
        # y = enc_ftrs[-1]
        # y = enc_ftrs
        y = y.permute([0,2,1])
        
        _,(self.h,self.c)=self.lstm( y , (self.h,self.c) )
        # C = self.c.permute([1,0,2]).repeat(1,y.shape[1],1)
        # y = torch.cat((self.h, self.c),2)
        y = self.h
        y = torch.squeeze(y)
        y=self.linear1(y)
        y=self.relu(y)
        # y=self.do(y)
        y=self.linear2(y)
        # y=self.relu(y)
        # y=self.sigm(y)
        # y = F.softmax(y, dim=1)

        return y
    
    def init_hiden(self,batch):
        self.h=torch.zeros((self.lstm_layers, batch, self.lstm_h_size)).cuda()
        self.c=torch.zeros((self.lstm_layers, batch, self.lstm_h_size)).cuda()
    

# class ClassGen(nn.Module):
#     def __init__(self, h_size=256,  num_class=1):
#         super().__init__()
#         self.linear1     = nn.Linear(h_size, h_size, bias=True)
#         self.do          = nn.Dropout(p=0.5)
#         self.linear2     = nn.Linear(h_size, num_class, bias=True)    
#         self.relu  = nn.ReLU()
#         # self.sigm = nn.Sigmoid()

#     def forward(self, x):
        
#         # x = x.permute([0,2,1])
#         x = torch.squeeze(x)
#         x=self.linear1(x)
#         x=self.relu(x)
#         # x=self.do(x)
#         x=self.linear2(x)
#         x=self.relu(x)
        
#         return x   
    
    
def CreateDataset(path_data):      

    h5_list = glob.glob( os.path.normpath( path_data + "**/*.h5"))  
    sigs_list = []
    lbl_ist = []
 
    for file_path in h5_list:
        f = h5py.File(file_path)
        for a in f.__iter__():
            sigs_list.append({'file_path': file_path, 'tname': a})
            # lbl_ist.append(np.asarray(f[a]['st']))
            lbl_ist.append(np.asarray(dictGen[file_path.split('\\')[-1].split('_')[0]]).astype(np.float32))
                
    return sigs_list, lbl_ist



dictGen = dict(gapA=0 , infB=1 , mdh=2 , pgi=3 , phoE=4 , rpoB=5 , tonB=6 )
 
path_data = 'C:\data\jakubicek/all_MLST_genes_new_format1'
dataset, lbl_list = CreateDataset(path_data)


batch = 64
classes = 2

net = GenNet( enc_chs=(1,64,128,256,512), lstm_h_size=512, h_size=512, num_class=classes).cuda()

optimizer = optim.Adam(net.parameters(), lr=0.0001,weight_decay=1e-6)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, verbose=True)

end_gen = 12530

train_list = dataset[0:end_gen:10]
# train_list = np.random.permutation( train_list )
lbl_list = lbl_list[0:end_gen:10]

w1,_= np.histogram(np.array(lbl_list), bins=np.arange(0,7+1,1)-0.5)
weight = torch.tensor((w1/np.sum(w1)).astype(np.float32) ).cuda()


train_loss = []
train_LOSS = []
train_acc = []
train_ACC = []


for epch in range(0,100):

    train_list = np.random.permutation( train_list )
    net.train()
    net.zero_grad()
    ii=0
    # train_acc=[]
    # Pred = torch.zeros((batch, 2, 1), dtype=torch.float32).cuda()
    # LBL = torch.zeros((batch, 1), dtype=torch.float32).cuda()

    for ite in range(0, len(train_list)-batch, batch):

        net.zero_grad()
        # net2.zero_grad()
        
##################### 
        # Pred = torch.zeros((batch, classes, 1), dtype=torch.float32).cuda()
        # Sig = torch.zeros((batch, 1, 50000), dtype=torch.float32).cuda().type(torch.long)
        # Lbl = torch.zeros((batch, 1), dtype=torch.float32).cuda().type(torch.long)
        
        # for b in range(0, batch,1):
        #     # net.zero_grad()
        #     net.init_hiden(1)
        #     file = train_list[ite+b]
        #     sig, lbl = loaders.Load_whole_signal_h5(file, dictGen)
    
        #     pred = net(sig.cuda())
        #     net.zero_grad()
        #     pred = F.softmax(pred, dim=2)
        #     pred = pred.permute([0,2,1])
            
        #     Pred[b,:,:] = pred
        #     Lbl[b,:] = torch.tensor(lbl)
            
        #     # # torch.tensor(np.expand_dims(np.expand_dims(lbl,0),0)).cuda().type(torch.long)
            
#####################

#######################
        # Sig, Lbl = loaders.Load_cut_signal_h5(ite, batch, train_list, dictGen) 
        Sig, Lbl = loaders.Load_cut_gen_h5(ite, batch, train_list, dictGen)

        net.init_hiden(batch)
        Pred = net(Sig.cuda())
        
        # net.init_hiden(batch)
        # Feat = net(Sig.cuda())
        # Pred = net2(Feat.cuda())
        Pred = F.softmax(Pred, dim=1)
        # Pred = F.sigmoid(Pred)
        # # Pred = Pred.permute([1,2,0])
####################3
      
        # w1,_= np.histogram(np.array(lbl_list), bins=np.arange(0,7+1,1)-0.5)
        # weight = torch.tensor((w1/np.sum(w1)).astype(np.float32) ).cuda()
        weight =torch.tensor( (0.6, 0.4) ).cuda()
        
        # loss = nn.CrossEntropyLoss(weight)( Pred,   Lbl ) 
        # loss = nn.CrossEntropyLoss(weight)( Pred,   torch.squeeze(Lbl.cuda().type(torch.long)) ) 
        # loss = nn.CrossEntropyLoss(weight[0:2])( Pred.squeeze(),   torch.squeeze(Lbl.cuda().type(torch.long)) ) 
        # loss = nn.CrossEntropyLoss()( Pred.squeeze(),   torch.squeeze(Lbl.cuda().type(torch.long)) ) 
        loss = nn.CrossEntropyLoss(weight)( Pred.squeeze(),   torch.squeeze(Lbl.cuda().type(torch.long)) )

        # one_hot = torch.nn.functional.one_hot(Lbl.squeeze().type(torch.long), 7)
        # loss = -torch.mean( torch.log(Pred[one_hot==1]))
        
        # loss = nn.CrossEntropyLoss()( gg,   torch.squeeze(Lbl.cuda().type(torch.long)) )
        
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
        # nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)
        optimizer.step()
        # scheduler.step()
        
        # optimizer2.zero_grad()
        # # loss.backward()
        # # nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
        # # nn.utils.clip_grad_value_(net2.parameters(), clip_value=2.0)
        # optimizer2.step()
        # # scheduler.step()

        net.zero_grad()
        # net2.zero_grad()
        torch.cuda.empty_cache()
        
        acc = (Lbl.detach().cpu().numpy().squeeze() == (Pred.detach().cpu().numpy().squeeze().argmax(1))).astype(np.dtype(float))
        
        train_loss.append(loss.detach().cpu().numpy())
        train_acc.append( np.mean(acc) )
        
        # train_ACC.append(np.mean(train_acc))
        
        if ii%(int((len(train_list))/batch/2))  == 0:
        # if ii%10 == 0: 
            
            train_LOSS.append(np.mean(train_loss))
            plt.figure
            plt.plot( train_LOSS )
            # plt.plot( train_loss )
            # plt.ylim([0, 1.0])
            plt.show()
            
            train_ACC.append(np.mean(train_acc))
            plt.figure
            plt.plot(train_ACC)
            # plt.ylim([0.0,1])
            plt.show()

            train_acc = []
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
            
            # t = time.time()
        ii=ii+1
          
    # train_ACC.append(np.mean(train_acc))
    
    # plt.figure
    # plt.plot(train_ACC)
    # plt.ylim([0.0,1])
    # plt.show()
        
    scheduler.step()


# batch=8
# proc=0.95


# path_data = os.path.normpath( 'C:\data\jakubicek\GEN_Data_reload')
# sigs_list = glob.glob(os.path.normpath( path_data + "\*.npy"))

# N =  np.array( np.shape(sigs_list))
# sigs_list = np.random.permutation( sigs_list )

# train_list = sigs_list[0:int(np.round(N*proc))]
# test_list = sigs_list[int(np.round(int(N)*proc))+1:int(N)]


# # # LSTM training○

# # net = NetGEN().cuda()
# net = torch.load(r"D:\jakubicek\Bioinformatika\netv5_0.pt")

# # net = torch.load(r"D:\jakubicek\Bioinformatika\netv3_0.pt")
# # net = torch.load(r"D:\jakubicek\Bioinformatika\netv2_0.pt")

# optimizer = optim.Adam(net.parameters(), lr=0.00001,weight_decay=1e-6)
# # optimizer = optim.SGD(net.parameters(), lr=0.0001, weight_decay=1e-6)
# # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1*3593/batch, gamma=0.1, verbose=False)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1, verbose=False)
# # net.init_hiden(batch)

# train_loss = []
# train_acc = []
# train_ACC = []

# for epch in range(0,20):
#     net.train()
#     ii=0
#     iii=1
#     indx=0
#     train_list = np.random.permutation( train_list )
    
#     # net.init_hiden(batch)
    
#     for ite in range(0, len(train_list), 1):
#     # for ite in range(0, 10, batch):
#         net.train()
#         net.zero_grad()
#         batch=8
        
#         if iii==1:
#             sample,lbl = loaders.loaderWinGen(indx, train_list, batch, mode='interp')
#             indx = indx+batch
#         elif iii==2:   
#             sample,lbl = loaders.loaderWinRand(indx, train_list, batch, mode='interp')
#             indx = indx+batch
#         elif iii==3:
#             sample,lbl = loaders.loaderWhole(indx, train_list, 1)
#             iii=0
#             indx = indx+1
#             batch=1

#         net.init_hiden(batch)
        
#         pred = net(sample.cuda())
#         pred = F.softmax(pred, dim=2)             

#         lbl = lbl.permute([0,2,1]).cuda()
#         lbl = F.interpolate(lbl, ( pred.shape[1]))
#         lbl = lbl[:,0,:]
#         # lbl = lbl.permute([0,2,1])
#         pred = pred.permute([0,2,1])
        
#         # weight = torch.tensor((0.05, 0.95)).cuda()
#         w1 =  (torch.sum(lbl[0,:])+0.0001) / (lbl.shape[1] +0.0001) 
#         weight = torch.tensor((w1, 1-w1)).cuda()
#         loss = nn.CrossEntropyLoss(weight)( pred,  lbl.type(torch.long) ) 

#         GT = lbl.detach().cpu().numpy()
#         P = pred[:,1,:].detach().cpu().numpy()>0.5
#         train_acc.append( np.mean( np.sum( GT==P , 1) / GT.shape[1] )  )
        
#         train_loss.append(loss.detach().cpu().numpy())
     
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # scheduler.step()

#         torch.cuda.empty_cache()
        
#         if ii%(int((len(train_list)/batch)/10))  == 0:
            
#             train_ACC.append(np.mean(train_acc))
            
#             # plt.figure
#             # plt.plot(train_loss)
#             # plt.ylim([0, 1.0])
#             # plt.show()
            
#             plt.figure
#             plt.plot(-np.log(train_ACC))
#             # plt.ylim([0.0,1])
#             plt.show()
            
#             plt.figure
#             plt.plot(lbl.detach().cpu().numpy()[0,:])
#             plt.plot(pred.detach().cpu().numpy()[0,1,:])
#             # plt.plot(P[0,:])
#             plt.ylim([0.0,1])
#             plt.show()    

#             train_acc = []                 
            
#         ii=ii+1
#         iii=iii+1
        
#         if indx+8 > len(train_list):
#             break
        
#     scheduler.step()

#     # n=0   
#     # for m in range(0, len(test_list), 1):
           
#     #     net.init_hiden(batch)
#     #     net.train(mode=False)
#     #     # net.zero_grad()
        
#     #     sample,lbl = loaderWin(m, test_list, batch )
        
#     #     pred = net(sample.cuda())
#     #     net.zero_grad()
          
#     #     pred = F.softmax(pred, dim=2) 
    
#     #     lbl = lbl.permute([0,2,1]).cuda()
#     #     lbl = F.interpolate(lbl, ( pred.shape[1]))
#     #     lbl = lbl[:,0,:]
    
#     #     pred = pred.permute([0,2,1])
        
#     #     # loss = nn.CrossEntropyLoss(weight=torch.tensor((0.1, 0.9)).cuda() )( pred,  lbl.type(torch.long) )
    
#     #     GT = lbl.detach().cpu().numpy()
#     #     P = pred[:,1,:].detach().cpu().numpy()>0.5
#     #     test_acc.append( np.mean( np.sum( GT==P , 1) / GT.shape[1] ) )
    
#     #     torch.cuda.empty_cache()
        
#     #     if n%100 == 0:
            
#     #         plt.figure
#     #         plt.plot(test_acc)
#     #         plt.ylim([0.0,1])
#     #         plt.show()
            
#     #         # plt.figure
#     #         # plt.plot(lbl.detach().cpu().numpy()[0,:])
#     #         # plt.plot(pred.detach().cpu().numpy()[0,1,:])
#     #         # # plt.plot(P[0,:])
#     #         # # plt.ylim([0.7,1])
#     #         # plt.show()        
            
#     #     n=n+1
        
#     torch.cuda.empty_cache()



    
# class Sig_Loader(Dataset): 
#     super().__init__()
#     def __init__(self, path_data):      
#         path_data = os.path.normpath(path_data)
#         sigs_list = glob.glob(os.path.normpath( path_data + "**/*.h5"))
        
#         # self.sigs_list = []
#         # for index in ind:
#         #       self.sigs_list.append(sigs_list[index])
        
#         for h5dataset_fp in sigs_list:
#             self._add_data_infos(str(h5dataset_fp.resolve()))
        
        
#     def _add_data_infos(self, file_path):
#         with h5py.File(file_path) as h5_file:
#             # Walk through all groups, extracting datasets
#             for gname, group in h5_file.items():
#                 for dname, ds in group.items():        
#                     self.data_info.append()
                        
#     def __len__(self):
#         return len(self.sigs_list)
    
#     def __getitem__(self, index):       
#         sig, lbl = load_H5(sig_path, index)
#         return sig, lbl

   
# dataset = Sig_Loader(path_data)   

# train_loader = DataLoader(dataset, shuffle=True, batch_size=batch)

