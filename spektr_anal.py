# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 10:39:45 2021

@author: jakubicek
"""

# spektrasl analysis

import scipy.signal as SSig
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob


path_data = 'C:\data\jakubicek\GEN_Data_reload'
path_data = os.path.normpath(path_data)
sigs_list = glob.glob(os.path.normpath( path_data + "\*.npy"))
        
N=[]
M=[]
I=[]
# for index in range(0,len(sigs_list)):
for index in range(0,1):
    sig, loc = np.load( sigs_list[index], allow_pickle=True )  
    loc.sort()  
    N = len(sig)
    lbl = np.zeros([N,1], dtype=bool)
    lbl[loc[0]:loc[1],0] = True
    lbl =  np.float32(lbl)
    

f, t, Sxx = SSig.spectrogram(sig, nperseg=200, mode='magnitude', nfft=500, noverlap=100)  

plt.figure()
plt.imshow(Sxx[0:100,:])
plt.show()

plt.figure
plt.plot(lbl)
 
# plt.hist(N,512)
# plt.show()          

# plt.hist(M,512)
# plt.show() 

