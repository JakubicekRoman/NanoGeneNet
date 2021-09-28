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


def crop_sig(sig, N, loc):

    ind = np.random.randint(loc[0]-N+1)
    z = loc[1]-N
    csig = sig[range(z,loc[0]),:]
    return csig