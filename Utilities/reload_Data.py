import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


path_data = "D:/Projekty/Bioinformatika/Data"



path_data = os.path.normpath(path_data)
genoms_list = glob.glob(os.path.normpath( path_data + "\*.npy"))


for ind, gen in enumerate(genoms_list):
    path = gen
    path = os.path.normpath(path)
    a = np.load(path, allow_pickle=True)
    
    df = pd.read_csv( path[0:-11] + 'region.csv' )


    for i in range(0,a.shape[0],1):
        loc = [ df.get('start')[i] ,  df.get('end')[i] ]
        np.save('Data_reload/' + 'Gen_' + str(ind) + '_sig_' + str(i) + '.npy', [a[i], loc])






# a = np.load('D:\Projekty\Bioinformatika\signals.npy',allow_pickle=True)

# # with open('D:\Projekty\Bioinformatika\usek.csv') as csv_file:
# #     csv_reader = csv.reader(csv_file, delimiter=',')
    
# # f = open("D:\Projekty\Bioinformatika\usek.csv", 'w')
# # writer = csv.writer(f)

# df = pd.read_csv('D:/Projekty/Bioinformatika/usek.csv')

# Sig = np.zeros([10000,])

# N = np.shape(df)[0];
# # N = 20

# for i in range(0,N,1):
#     loc = [ df.get('start')[i] ,  df.get('end')[i] ]
#     loc.sort()
#     sig = a[i][loc[0]:loc[1]]
    
#     Sig = Sig + (np.interp( np.linspace(0,  np.shape(sig)[0]-1 , 10000 ), np.linspace(0,np.shape(sig)[0]-1,np.shape(sig)[0]), sig )/N)
#     # plt.figure()
#     # plt.ylim([40,120])
#     # plt.plot(Sig)
    
    
# plt.figure()
# # plt.ylim([40,120])
# plt.plot(Sig)

# R = abs(np.convolve(a[1] - np.mean(sig), Sig[::-1] - np.mean(Sig), mode='valid'))

# plt.figure()
# plt.plot(R)
