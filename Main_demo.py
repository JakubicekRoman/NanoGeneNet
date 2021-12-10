# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:28:21 2021

@author: jakubicek
"""

import os
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

from Def_NanoGeneNet import NetGEN
from Def_NanoGeneNet import ClassGEN
from Def_NanoGeneNet import Block
from Def_NanoGeneNet import Encoder

from Load import Load_signal_h5

dictGen = dict(gapA=0 , infB=1 , mdh=2 , pgi=3 , phoE=4 , rpoB=5 , tonB=6, run=7)


###########   Load nets   ###########
net1 = torch.load(os.path.normpath( os.getcwd() + "\Models\\net_det.pt" ) )
net2 = torch.load(os.path.normpath( os.getcwd() + "\Models\\net_class.pt") )



###########   Load squiggle, h5 format     ###########
# squiggle_path = os.path.normpath( os.getcwd() + "\Data_example\\readExample.h5" )
squiggle_path = os.path.normpath( os.getcwd() + "\Data_example\\phoE_KP1037_signals_in_barcode10.h5" )

squiggle, lbl, clbl, loc = Load_signal_h5(squiggle_path, dictGen)


###########  Prediction  ###########
net1.train(mode=False) 
net2.train(mode=False) 

with torch.no_grad():
    net1.init_hiden(1)            
    pred, feat = net1(squiggle.cuda())
    pred = F.softmax(pred, dim=2)
    
    feat = torch.cat((feat[:,:,0:256], torch.unsqueeze( pred[:,:,1],2 ) ),2)
    
    net2.init_hiden(1)
    pred2 = net2(feat.cuda())
    pred2 = F.softmax(pred2, dim=0)
    


###########  Display  ###########
downsamplig = 16
loc = (loc/downsamplig).astype( np.int64 )

lbl = lbl.permute([0,2,1]).cuda()
lbl = F.interpolate(lbl, ( pred.shape[1]))
lbl = lbl[:,0,:]

s = squiggle.permute([0,2,1]).cuda()
s = F.interpolate(s, ( pred.shape[1]))
s = s[:,0,:]
s = s[0,:].detach().cpu().numpy()

plt.figure
plt.plot(s)
plt.plot(np.arange( int(loc[0]), int(loc[1])), s[int(loc[0]):int(loc[1])], color='red')
# plt.ylim([0.5, 1.0])  
plt.show() 
        
plt.figure
plt.plot(pred[0,:,1].detach().cpu().numpy())
plt.plot(lbl[0,:].detach().cpu().numpy())

print( "Gene number: " + str(int(clbl.detach().cpu().numpy()[0])) + "\nPredicted gene nnumber: " + str(int(pred2.detach().cpu().numpy().argmax(0) )) )




