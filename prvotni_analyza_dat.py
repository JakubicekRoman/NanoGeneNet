# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 17:57:42 2021

@author: jakubicek
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob


path_data = 'C:\Data\Jakubicek\Bioinformatika\Data_reload'
path_data = os.path.normpath(path_data)
sigs_list = glob.glob(os.path.normpath( path_data + "\*.npy"))
        
N=[]
M=[]
I=[]
for index in range(0,len(sigs_list)):
    sig, loc = np.load( sigs_list[index], allow_pickle=True )  
    N.append(len(sig))
    loc.sort()  
    M.append(loc[1]-loc[0])
    if N[-1]>2000000:
        I.append(index)
        print( sigs_list[index])

 
plt.hist(N,512)
plt.show()          

plt.hist(M,512)
plt.show() 

# a = np.load('D:\Projekty\Bioinformatika\Data\with_gapA\gapA_3KP_FCvsFL_RBK_barcode08_KP1210_signals.npy',allow_pickle=True)

# # with open('D:\Projekty\Bioinformatika\usek.csv') as csv_file:
# #     csv_reader = csv.reader(csv_file, delimiter=',')
    
# # f = open("D:\Projekty\Bioinformatika\usek.csv", 'w')
# # writer = csv.writer(f)

# df = pd.read_csv('D:/Projekty/Bioinformatika/Data/with_gapA/gapA_3KP_FCvsFL_RBK_barcode08_KP1210_region.csv')

# Sig = np.zeros([10000,])

# N = np.shape(df)[0];
# # N = 20

# for i in range(0,N,1):
#     loc = [ df.get('start')[i] ,  df.get('end')[i] ]
#     loc.sort()
#     sig = a[i][loc[0]:loc[1]]
    
#     # Sig = Sig + (np.interp( np.linspace(0,  np.shape(sig)[0]-1 , 10000 ), np.linspace(0,np.shape(sig)[0]-1,np.shape(sig)[0]), sig )/N)
#     # plt.figure()
#     # plt.ylim([40,120])
#     # plt.plot(Sig)
    
    
# plt.figure()
# # plt.ylim([40,120])
# plt.plot(Sig)

# R = abs(np.convolve(a[1] - np.mean(sig), Sig[::-1] - np.mean(Sig), mode='valid'))

# plt.figure()
# plt.plot(R)


# # b = a[0][17664:22224]
# # b2 = a[1][153974:161848]

# # plt.subplot(2,1,1)
# # plt.ylim([40,120])
# # plt.plot(np.linspace( 0, np.shape(b2)[0]-1, np.shape(b)[0]) , b)
# # plt.subplot(2,1,2)
# # plt.plot(np.linspace( 0, np.shape(b2)[0]-1, np.shape(b2)[0]) , b2)


