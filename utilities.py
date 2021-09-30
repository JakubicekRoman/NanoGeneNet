# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 18:39:04 2021

@author: jakubicek
"""
import torch
import numpy as np
 
def dice_loss(X, Y):
    eps = 1.
    dice = ((2. * torch.sum(X*Y) + eps) / (torch.sum(X) + torch.sum(Y) + eps) )
    return 1 - dice


def crop_sig(sig, loc):

    # ind = np.random.randint(loc[0]-N+1)
    # z = loc[1]-N
    z = loc[0]-10000
    k = loc[1]+10000
    if z<0:
        z=0
    if k>sig.shape[0]:
        k=sig.shape[0]
        
    sig = sig[range(int(z),int(k)),:].copy()
    
    vel = sig.shape[1]
    
    N = 8000
    newsig = np.zeros((N,vel))

    for i in range(0,vel):
        newsig[:,i] = np.interp(np.linspace(0,sig.shape[0]-1,N), np.linspace(0,sig.shape[0]-1,sig.shape[0]), sig[:,i])  
    
    return newsig    
    return newsig