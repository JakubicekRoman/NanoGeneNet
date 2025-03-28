import os
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import h5py
from scipy.io import savemat
import time

from Def_NanoGeneNet import NetGEN
from Def_NanoGeneNet import ClassGEN
from Def_NanoGeneNet import Block
from Def_NanoGeneNet import Encoder

from Load import Load_signal_h5
from Utilities import utilities

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

dictGen = dict(gapA=0 , infB=1 , mdh=2 , pgi=3 , phoE=4 , rpoB=5 , tonB=6, run=7)


###########   Load nets   ###########
net1 = torch.load(os.path.normpath( os.getcwd() + "\Models\\net_det.pt" ) , map_location=torch.device('cpu'))
net2 = torch.load(os.path.normpath( os.getcwd() + "\Models\\net_class.pt") , map_location=torch.device('cpu'))

net1 = net1.to(device)
net2 = net2.to(device)
net1.init_hiden(1)
net2.init_hiden(1)
net1.h = net1.h.to(device)
net1.c = net1.c.to(device)
net2.h = net2.h.to(device)
net2.c = net2.c.to(device)

dice_val=[]

###########   Load squiggle, h5 format     ###########
# squiggle_path = os.path.normpath( os.getcwd() + "\Data_example\\readExample.h5" )
squiggle_path = os.path.normpath( os.getcwd() + "\Data_example\\phoE_KP1037_signals_in_barcode10.h5" )
# squiggle_path = os.path.normpath( os.getcwd() + "\data\\FAQ14273_pass_barcode02_903a7c29_0.fast5" )


# squiggle, lbl, clbl, loc = Load_signal_h5(squiggle_path, dictGen)

f = h5py.File(squiggle_path,'r')
reads = list(f)

print("Number of reads in file: ", len(reads))

pred_sig = []
lbl = []

# time measurement
start = time.time()

for i in range(0,len(reads)):
# for i in range(0,10):
    ###########  Load squiggle  ###########
    read = f['/'][reads[i]]
    squiggle = read['signal'][()]
    squiggle = np.expand_dims(squiggle,0)
    squiggle = torch.tensor(np.expand_dims(squiggle,2))
    
    ###########  Prediction  ###########
    net1.train(mode=False) 
    net2.train(mode=False) 
    
    with torch.no_grad():
        # net1.init_hiden(1)            
        pred, feat = net1(squiggle.to(device))
        pred = F.softmax(pred, dim=2)
        
        feat = torch.cat((feat[:,:,0:256], torch.unsqueeze( pred[:,:,1],2 ) ),2)
        
        # net2.init_hiden(1)
        pred2 = net2(feat.to(device))
        pred2 = F.softmax(pred2, dim=0)
        
    
    pred = pred.permute([0,2,1])
    pred = F.interpolate(pred, ( squiggle.shape[1]))
    pred = pred[0,1,:].detach().cpu()
    
    # lbl = int(pred2.detach().cpu().numpy().argmax(0) )
    lbl.append(int(pred2.detach().cpu().argmax(0) ))
    
    # print(str(i/len(reads)*100))
    print(str(np.shape(squiggle)[1]))

x = np.array(lbl)
print(np.unique(x))

# time measurement
end = time.time()
print("Time taken: ", end-start)
# cas for one read:
print("Time taken for one read: ", (end-start)/len(reads))

    # pred_sig.append(pred)
    # lbl.append(int(pred2.detach().cpu().numpy().argmax(0) ))
    
    # savedict = {
    #     'Pred_sig' : pred,
    #     'Lable' : lbl,
    # }
    # savemat('read_'+ f"{i:04d}"+'.mat', savedict)

# savemat('file.mat', mdict={'list':pred_sig})

# savemat('file.mat', {'cell' : pred_sig})
            
# savemat('predikce.mat', {'pred_sig':pred_sig,'lables':lbl})


# import numpy
# import scipy.io
# pred_sig = ['hello', 'world!!!']
# savemat('file.mat', mdict={'list':pred_sig})

# import scipy.io as sio
# # **** data
# A = np.random.randint(0,10,1000)
# B = np.random.randint(0,10,100)
# # collect arrays in dictionary
# savedict = {
#     'A' : A,
#     'B' : B,
# }
# # save to disk
# sio.savemat('my_arrays.mat', savedict)

# ###########  Display  ###########

# pred = pred.permute([0,2,1])
# pred = F.interpolate(pred, ( squiggle.shape[1]))
# pred = pred[0,1,:].detach().cpu().numpy()

# plt.figure
# plt.plot(pred)
# plt.ylim([0, 1.0])

# print( "\nPredicted gene nnumber: " + str(int(pred2.detach().cpu().numpy().argmax(0) )) )



