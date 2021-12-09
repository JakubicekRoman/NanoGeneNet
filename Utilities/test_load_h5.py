# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:35:18 2021

@author: jakubicek
"""
import load_h5_Def as h5
from torch.utils import data
import os


loader_params = {'batch_size': 1, 'shuffle': True, 'num_workers': 6}


# path_data = os.path.normpath( 'C:/data/jakubicek/gapA_new_format')
path_data = 'C:/data/jakubicek/gapA_new_format'

dataset = h5.HDF5Dataset(path_data, recursive=True, load_data=False, data_cache_size=1, transform=None)


# data_loader = data.DataLoader(dataset, **loader_params)

