# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 20:23:14 2021

@author: jakubicek
"""

import torch.nn as nn
import torch

rnn = nn.LSTM(100, 20, 2)
input = torch.randn(5, 3, 100)

h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))

res  = output.detach().cpu().numpy()

print(rnn)
