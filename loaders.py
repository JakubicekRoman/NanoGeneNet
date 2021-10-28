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
    
    sig = np.interp(np.linspace(0,N-1,int(N/4)), np.linspace(0,N-1,N), sig).astype(np.float32)
    
    lbl = np.asarray(dictGen[path.split('\\')[-1].split('_')[0]]).astype(np.float32)
    
    sig = np.expand_dims(sig,0)
    sig = torch.tensor(np.expand_dims(sig,2))

    # lbl = torch.tensor(np.expand_dims(lbl,1))
    
    return sig, lbl


def Load_cut_signal_h5(ite, batch, train_list, dictGen) :
    
    n = 50000
    Lbl = torch.tensor(np.zeros((batch,1), dtype=np.float32))
    Sig = torch.tensor(np.zeros((batch,int(n),1), dtype=np.float32))
    
    
    for i in range(0,batch):
        file = train_list[ite+i]
        a = file['tname']
        path = file['file_path']
        f = h5py.File(path,'r')
        sig = np.asarray(f[a]['signal']).astype(np.float32)
        loc = np.asarray(f[a]['coord']).astype(np.float32)
        loc.sort()
        N = len(sig)
        
        if N < n:
            sig_ = np.interp(np.linspace(0,N-1,n), np.linspace(0,N-1,N), sig)
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
        
        sig = np.expand_dims(sig_, 0)
        sig = torch.tensor(np.expand_dims(sig,2))
        
        lbl = np.asarray(dictGen[path.split('\\')[-1].split('_')[0]]).astype(np.float32)
        
        Sig[i,:,:] = sig
        Lbl[i,:] = torch.tensor(lbl)    
    
    return Sig, Lbl



def Load_cut_gen_h5(ite, batch, train_list, dictGen) :   
    n = 10000
    Lbl = torch.tensor(np.zeros((batch,1), dtype=np.float32))
    Sig = torch.tensor(np.zeros((batch,int(n),1), dtype=np.float32))
    
    
    for i in range(0,batch):
        file = train_list[ite+i]
        a = file['tname']
        path = file['file_path']
        f = h5py.File(path,'r')
        loc = np.asarray(f[a]['coord']).astype(np.float32)
        loc.sort()
        sig = np.asarray(f[a]['signal']).astype(np.float32)
        
        if loc[1]==0:
            sig = np.zeros(n)
            loc[1]=n

        sig = sig[int(loc[0]):int(loc[1])]
        N = len(sig)
        sig = np.interp(np.linspace(0,N-1,n), np.linspace(0,N-1,N), sig)
        
        sig = np.expand_dims(sig, 0)
        sig = torch.tensor(np.expand_dims(sig,2))
        
        lbl = np.asarray(dictGen[path.split('\\')[-1].split('_')[0]]).astype(np.float32)
        
        Sig[i,:,:] = sig
        Lbl[i,:] = torch.tensor(lbl)    
    
    return Sig, Lbl





# nacteni signalu a okna vzdy obsahujici gen
def loaderWinRand(ite, train_list, batch, mode='interp'):
 
    # n = 100000
    n = random.randrange(20000, 80000)
    nfin = n + random.randrange(int(-0.1*n), int(0.1*n))
    
    LBL = torch.tensor(np.zeros((batch,int(nfin),2), dtype=np.float32))
    Sig = torch.tensor(np.zeros((batch,int(nfin),1), dtype=np.float32))
    
    for i in range(0,batch):
        sig, loc = np.load( train_list[ite+i], allow_pickle=True )
        N = len(sig)
        loc.sort()
        lbl = np.zeros([N,1], dtype=bool)
        lbl[loc[0]:loc[1],0] = True
        lbl =  np.float32(lbl)        
        if N < n:
            if mode == 'interp':
                sig_ = np.interp(np.linspace(0,N-1,n), np.linspace(0,N-1,N), sig)
                lbl_ = np.interp(np.linspace(0,N-1,n), np.linspace(0,N-1,N), lbl[:,0])
            elif mode=='pad':
                sig_ = np.pad(sig, [0,n-N], mode='constant')
                lbl_ = np.pad( lbl[:,0], [0,n-N], mode='constant')
            n = N-1
        M = random.randrange(0, N-n)
        sig_ = sig[range(int(M),int(M)+n)]
        lbl_ = lbl[range(int(M),int(M)+n)]
            
        sig_ = np.interp(np.linspace(0,n-1,nfin), np.linspace(0,n-1,n), sig_)
        lbl_ = np.interp(np.linspace(0,n-1,nfin), np.linspace(0,n-1,n), lbl_[:,0])
            
        lbl = np.zeros([nfin,2]).astype(np.float32)
        lbl[:,0] = lbl_.T
        lbl[:,1] = (~lbl_[:].astype(np.bool_)).astype(np.float32).T
        sig = np.expand_dims(sig_ ,1)
            
        
        Sig[i,:,:] = torch.tensor(sig.astype(np.float32))
        LBL[i,:,:] = torch.tensor(lbl.astype(np.float32))

    return Sig, LBL



# nacteni vzreyu signalu ciste nahodne
def loaderWinGen(ite, train_list, batch, mode='interp'):

    n = 50000
    # n = random.randrange(20000, 100000)
    LBL = torch.tensor(np.zeros((batch,int(n),2), dtype=np.float32))
    Sig = torch.tensor(np.zeros((batch,int(n),1), dtype=np.float32))
    
    for i in range(0,batch):
        sig, loc = np.load( train_list[ite+i], allow_pickle=True )
        N = len(sig)
        loc.sort()
        lbl = np.zeros([N,1], dtype=bool)
        lbl[loc[0]:loc[1],0] = True
        lbl =  np.float32(lbl)    
        
        if N < n:
            if mode == 'interp':
                sig_ = np.interp(np.linspace(0,N-1,n), np.linspace(0,N-1,N), sig)
                lbl_ = np.interp(np.linspace(0,N-1,n), np.linspace(0,N-1,N), lbl[:,0])
            elif mode=='pad':
                sig_ = np.pad(sig, [0,n-N], mode='constant')
                lbl_ = np.pad( lbl[:,0], [0,n-N], mode='constant')
        else:
            z = loc[1]-n
            if z<0:
                z=0
            k=loc[0]-1
            if k+n  >sig.shape[0]:
                k=sig.shape[0]-n-1
    
            M = random.randrange(z, k)
            sig_ = sig[range(int(M),int(M)+n)]
            lbl_ = lbl[range(int(M),int(M)+n)]
            
        lbl = np.zeros([n,2]).astype(np.float32)
        lbl[:,0] = lbl_.T
        lbl[:,1] = (~lbl_[:].astype(np.bool_)).astype(np.float32).T
        sig = np.expand_dims(sig_ ,1)
            
        Sig[i,:,:] = torch.tensor(sig.astype(np.float32))
        LBL[i,:,:] = torch.tensor(lbl.astype(np.float32))
    
    return Sig, LBL




  # nacteni celzch signalu
def loaderWhole(ite, train_list, batch=1):
    sig, loc = np.load( train_list[ite], allow_pickle=True )
    N = len(sig)
    loc.sort()
    lbl = np.zeros([N,1], dtype=bool)
    lbl[loc[0]:loc[1],0] = True
    lbl =  np.float32(lbl)

    lbl_ = lbl[:,0]
    lbl = np.zeros([N,2]).astype(np.float32)
    lbl[:,0] = lbl_.T
    lbl[:,1] = (~lbl_[:].astype(np.bool_)).astype(np.float32).T
    
    sig = np.expand_dims(sig ,1)    
    sig = np.expand_dims(sig,0)
    lbl = np.expand_dims(lbl,0)
    
    return torch.tensor(sig.astype(np.float32)), torch.tensor(lbl.astype(np.float32))