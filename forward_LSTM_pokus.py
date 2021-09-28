import os
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
# import matplotlib.image as im
# import nibabel as nib
import SimpleITK as sitk
# import medpy as medpy
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
    def __init__(self, path_data):      
        path_data = os.path.normpath(path_data)
        self.sigs_list = glob.glob(os.path.normpath( path_data + "\*.npy"))
               
    
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

    
batch=1

path_data = os.path.normpath( 'D:\Projekty\Bioinformatika\Data_reload')
dataset = nacitac(path_data)   
train_loader = DataLoader(dataset, shuffle=True, batch_size=batch)



# # LSTM test
# sig, lbl = dataset[0]

hiden_dim=50

net = LSTM(1, hiden_dim, 2).cuda()

# net.train()
net.eval()

net.init_hiden(batch)
net = net.cuda()

# for j,sig in enumerate(train_loader):   
    # y_pred = net(sig)

sig, lbl = dataset[0]
sig = np.matlib.repmat(sig,1,batch)
sig = np.expand_dims(sig.T,2)       
sig = torch.tensor(sig).cuda()

y_pred = net(torch.tensor(sig))
y_pred = F.softmax(y_pred, dim=2)


pred = y_pred.clone().detach().cpu().numpy().squeeze()
lbl = lbl.detach().cpu().numpy()

plt.figure
plt.plot(lbl[:,0])
plt.plot(pred[:,0])

