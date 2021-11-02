# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 18:39:04 2021

@author: jakubicek
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
 
def dice_loss_torch(X, Y):
    eps = 1.
    dice = ((2. * torch.sum(X*Y) + eps) / (torch.sum(X) + torch.sum(Y) + eps) )
    return 1 - dice



def dice_torch(X, Y):
    eps = 1.
    dice = ((2. * torch.sum(X*Y) + eps) / (torch.sum(X) + torch.sum(Y) + eps) )
    return dice



def WCE_loss(X, Y):
    w = np.sum(  np.array( Y[:,:,0]) ,1)  / Y.shape[1]
    w = torch.tensor(np.matlib.repmat(w, X.shape[1], 1).T).cuda()
    
    WCE = torch.mean( -torch.log( (  X[:,:,1]  \
                                   + X[:,:,0]  ) /2 ) )
    return WCE


def comp_class_acc(clbl, cacc):
    hi=np.zeros((8))
    for i in range(0,8):
        ind = (clbl == i)
        acc = np.mean( cacc[ind] )
        hi[i] = acc
        
    return hi




# def loader(ite,sigs_list,trainIND, batch):
#     # M = int(5000)
#     M1 = random.randrange(100, 10000)
#     # M2 = 5000 - M1 
#     M2 = random.randrange(100, 10000)
    
#     # LBL = torch.tensor(np.zeros((batch,int((M1+M2)/8),2), dtype=np.float32))
#     # Sig = torch.tensor(np.zeros((batch,int((M1+M2)/8),1), dtype=np.float32))
#     LBL = torch.tensor(np.zeros((batch,int(5000),2), dtype=np.float32))
#     Sig = torch.tensor(np.zeros((batch,int(5000),1), dtype=np.float32))
    
#     for i in range(0,batch):
#         sig, loc = np.load( sigs_list[ite+i], allow_pickle=True )
#         N = len(sig)
#         loc.sort()       
#         sig = sig.astype(np.float32)
#         sig = np.expand_dims(sig,1)
        
#         lbl = np.zeros([N,2], dtype=bool)
#         lbl[loc[0]:loc[1],0] = True
#         lbl[:,1] = ~lbl[:,0]
#         lbl =  np.float32(lbl)
#         # lbl =  torch.tensor( np.float32(lbl) )
                
#         sig = crop_sig(sig, loc, (M1,M2))
#         lbl = crop_sig(lbl, loc, (M1,M2))
#         # .astype(np.float32)
#         LBL[i,:,:] = torch.tensor(lbl)
#         Sig[i,:,:] = torch.tensor(sig)
#         # LBL[i,:,:] = lbl
#         # Sig[i,:,:] = sig
    
#     return  Sig, LBL
  
  

# def crop_sig(sig, loc, M):

#     # ind = np.random.randint(loc[0]-N+1)
#     # z = loc[1]-N
#     z = loc[0]-M[0]
#     k = loc[1]+M[1]
#     if z<0:
#         z=0
#     if k>sig.shape[0]:
#         k=sig.shape[0]
        
#     sig = sig[range(int(z),int(k)),:]
    
#     vel = sig.shape[1]
    
#     # N = int( np.sum(M)/8 )
#     N = int(5000)
#     newsig = np.zeros((N,vel))
#     # newsig=[];
    
#     # newsig = torch.nn.functional.interpolate(torch.unsqueeze(sig.transpose(1,0), 0).cuda(), (N) ).transpose(1,2)
#     # torch.nn.functional.interpolate(sig, (N) )

#     for i in range(0,vel):
#         newsig[:,i] = np.interp(np.linspace(0,sig.shape[0]-1,N), np.linspace(0,sig.shape[0]-1,sig.shape[0]), sig[:,i])  
    
#     return newsig    



# def loaderWin(ite, train_list, batch):
 

#     n = 50000
#     LBL = torch.tensor(np.zeros((batch,int(n),2), dtype=np.float32))
#     Sig = torch.tensor(np.zeros((batch,int(n),1), dtype=np.float32))
    
#     for i in range(0,batch):
#         sig, loc = np.load( train_list[ite+i], allow_pickle=True )
#         N = len(sig)
#         loc.sort()
#         lbl = np.zeros([N,1], dtype=bool)
#         lbl[loc[0]:loc[1],0] = True
#         lbl =  np.float32(lbl)
#         # sig = np.expand_dims(sig,1)
        
#         if N < n:
#             sig_ = np.interp(np.linspace(0,N-1,n), np.linspace(0,N-1,N), sig)  
#             lbl_ = np.interp(np.linspace(0,N-1,n), np.linspace(0,N-1,N), lbl[:,0])
#         else:      
#             z = loc[1]-n
#             if z<0:
#                 z=0
#             k=loc[0]-1
#             if k+n  >sig.shape[0]:
#                 k=sig.shape[0]-n-1

#             M = random.randrange(z, k)
#             sig_ = sig[range(int(M),int(M)+n)]
#             lbl_ = lbl[range(int(M),int(M)+n)]
            
#         lbl = np.zeros([n,2]).astype(np.float32)
#         lbl[:,0] = lbl_.T
#         lbl[:,1] = (~lbl_[:].astype(np.bool_)).astype(np.float32).T
#         sig = np.expand_dims(sig_ ,1)
#         # plt.plot(lbl)
#         # plt.show()
            
#         Sig[i,:,:] = torch.tensor(sig.astype(np.float32))
#         LBL[i,:,:] = torch.tensor(lbl.astype(np.float32))
    
#     return  Sig, LBL
