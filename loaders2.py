# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 09:29:11 2021

@author: jakubicek
"""

import numpy as np
import numpy.matlib
import torch
import random
import h5py



# nacteni h5 signalu
def load_H5(sig_path):
    f = h5py.File( sig_path , 'r')
    a = list(f.keys())[0]
    sig = np.asarray(f[a]['signal'])
    loc = list(f[a]['coord'])
    lbl = np.asarray(f[a]['st'])

    return sig, loc, lbl


def Load_whole_signal_h5(file, dictGen):
    
    a = file['tname']
    path = file['file_path']
    
    f = h5py.File(path,'r')
    sig = np.asarray(f[a]['signal']).astype(np.float32)
    
    N = len(sig)
    
    loc = np.asarray(f[a]['coord']).astype(np.float32)
    loc.sort()
    lbl = np.zeros([N,], dtype=bool)
    lbl[int(loc[0]):int(loc[1])] = True
    lbl =  np.float32(lbl)
    
    n = 100000
    if N>1000000:
        z = loc[1]-n
        if z<0:
            z=0
        k=loc[0]-1
        if k+n  >sig.shape[0]:
            k=sig.shape[0]-n-1
        if k<=z:
            z=0
            k=1
        M = random.randrange(int(z), int(k))
        sig = sig[range(int(M),int(M)+n)]
        lbl = lbl[range(int(M),int(M)+n)]
        
        # sig = np.interp(np.linspace(0,N-1,int(N/4)), np.linspace(0,N-1,N), sig).astype(np.float32)
        # lbl = np.interp(np.linspace(0,N-1,int(N/4)), np.linspace(0,N-1,N), lbl).astype(np.float32)
        
    # lbl = np.asarray(dictGen[path.split('\\')[-1].split('_')[0]]).astype(np.float32)
    
    sig = np.expand_dims(sig,0)
    sig = torch.tensor(np.expand_dims(sig,2))
    
    lbl = np.expand_dims(lbl,0)
    lbl = torch.tensor(np.expand_dims(lbl,2))
    
    clbl = torch.tensor(np.zeros((1), dtype=np.float32))
    clbl[0] = torch.tensor( np.asarray(dictGen[path.split('\\')[-1].split('_')[0]]).astype(np.float32) )
    

    
    # lbl = torch.tensor(np.expand_dims(lbl,1))
    
    return sig, lbl, clbl


def Load_cut_signal_h5(ite, batch, train_list, dictGen):
    
    # n = 50000
    # n = random.randrange(30000, 70000)
    n = random.randrange(20000, 80000)
    Lbl = torch.tensor(np.zeros((batch,int(n),1), dtype=np.float32))
    Sig = torch.tensor(np.zeros((batch,int(n),1), dtype=np.float32))
    Clbl = torch.tensor(np.zeros((batch), dtype=np.float32))
    
    for i in range(0,batch):
        file = train_list[ite+i]
        a = file['tname']
        path = file['file_path']
        f = h5py.File(path,'r')
        sig = np.asarray(f[a]['signal']).astype(np.float32)
        N = len(sig)
        loc = np.asarray(f[a]['coord']).astype(np.float32)
        loc.sort()
        lbl = np.zeros([N,], dtype=bool)
        lbl[int(loc[0]):int(loc[1])] = True
        lbl =  np.float32(lbl)

        
        if N < n:
            sig_ = np.interp(np.linspace(0,N-1,n), np.linspace(0,N-1,N), sig)
            lbl_ = np.interp(np.linspace(0,N-1,n), np.linspace(0,N-1,N), lbl)
        else:
            z = loc[1]-n
            if z<0:
                z=0
            k=loc[0]-1
            if k+n  >sig.shape[0]:
                k=sig.shape[0]-n-1
            if k<=z:
                z=0
                k=1
            M = random.randrange(int(z), int(k))
            sig_ = sig[range(int(M),int(M)+n)]
            lbl_ = lbl[range(int(M),int(M)+n)]
        
        sig = np.expand_dims(sig_, 0)
        sig = torch.tensor(np.expand_dims(sig,2))
        lbl = np.expand_dims(lbl_, 0)
        lbl = torch.tensor(np.expand_dims(lbl,2))
                
        Sig[i,:,:] = sig
        Lbl[i,:,:] = lbl         
        Clbl[i] = torch.tensor( np.asarray(dictGen[path.split('\\')[-1].split('_')[0]]).astype(np.float32))
    
    return Sig, Lbl, Clbl



# def Load_cut_gen_h5(ite, batch, train_list, dictGen) :   
#     n = 10000
#     Lbl = torch.tensor(np.zeros((batch,1), dtype=np.float32))
#     Sig = torch.tensor(np.zeros((batch,int(n),1), dtype=np.float32))
    
    
#     for i in range(0,batch):
#         file = train_list[ite+i]
#         a = file['tname']
#         path = file['file_path']
#         f = h5py.File(path,'r')
#         loc = np.asarray(f[a]['coord']).astype(np.float32)
#         loc.sort()
#         sig = np.asarray(f[a]['signal']).astype(np.float32)
        
#         if loc[1]==0:
#             sig = np.zeros(n)
#             loc[1]=n

#         sig = sig[int(loc[0]):int(loc[1])]
#         N = len(sig)
#         sig = np.interp(np.linspace(0,N-1,n), np.linspace(0,N-1,N), sig)
        
#         sig = np.expand_dims(sig, 0)
#         sig = torch.tensor(np.expand_dims(sig,2))
        
#         lbl = np.asarray(dictGen[path.split('\\')[-1].split('_')[0]]).astype(np.float32)
        
#         Sig[i,:,:] = sig
#         Lbl[i,:] = torch.tensor(lbl)    
    
#     return Sig, Lbl


