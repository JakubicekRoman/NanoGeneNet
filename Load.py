# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 16:25:14 2021

@author: jakubicek
"""

import numpy as np
import numpy.matlib
import torch
import h5py


def Load_signal_h5(path, dictGen):
    
    f = h5py.File(path,'r')
    a = list(f.keys())[3]
    sig = np.asarray(f[a]['signal']).astype(np.float32)
    
    N = len(sig)
    
    loc = np.asarray(f[a]['coord']).astype(np.float32)
    loc.sort()
    lbl = np.zeros([N,], dtype=bool)
    lbl[int(loc[0]):int(loc[1])] = True
    lbl =  np.float32(lbl)
    
    
    sig = np.expand_dims(sig,0)
    sig = torch.tensor(np.expand_dims(sig,2))
    
    lbl = np.expand_dims(lbl,0)
    lbl = torch.tensor(np.expand_dims(lbl,2))
    
    clbl = torch.tensor(np.zeros((1), dtype=np.float32))
    clbl[0] = torch.tensor( np.asarray(dictGen[path.split('\\')[-1].split('_')[0]]).astype(np.float32) )
    
    return sig, lbl, clbl, loc