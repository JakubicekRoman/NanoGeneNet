import os
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as im
# import nibabel as nib
import SimpleITK as sitk
# import medpy as medpy
# import pandas as pd
import torch
from torch.utils import data
import torch.optim as optim
import glob





class DataLoader():
    def __init__(self, path_data, pat):      
        path_data = os.path.normpath(path_data)
        self.pat_list = glob.glob(os.path.normpath( path_data + "\*raw.nii.gz"))
        self.data_list = []
        self.mask_list = []
        self.slice = []
        self.n_pat = []
        self.size_data=[]
        # print(self.pat_list[0])
        # self.name = glob.glob(self.pat_list[0] + "/*nii.gz")
        # print(' '.join(self.name[0]))
        
        # for i in pat:
        #     path = os.path.normpath(self.pat_list[i])
            
        #     file_reader = sitk.ImageFileReader()
        #     file_reader.SetFileName(path)
        #     file_reader.ReadImageInformation()
        #     size=file_reader.GetSize()
    
        #     ind = np.random.permutation(np.arange(20,size[2])) 
            
        #     for k in range(10):
        #         # for k in [0]:
        #         self.data_list = self.data_list + [os.path.normpath(path)]
        #         p =  os.path.normpath(path.replace("raw","mask",1))
        #         self.mask_list = self.mask_list + [p]
        #         self.slice = self.slice + [ind[k]]
        #         self.n_pat = self.n_pat + [i]
        #         self.size_data = self.size_data + [size]
        #         # print(' '.join(self.data_list[i]))
        #         # print(self.path)
                
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        
        # size = self.size_data[index]
        # size_cut = [224,224,1];
        # vel_maxTr = [np.maximum(size[0]-size_cut[0]-1,1), np.maximum(size[1]-size_cut[1]-1,1)]
        
        # transl = [np.random.randint(0,vel_maxTr[0]), np.random.randint(0,vel_maxTr[1]), int(self.slice[index])]

        # img = load_data.read_nii_position(self.data_list[index], size_cut, transl)
        # img = torch.tensor(((img.astype(np.float32))+1024)/4096)
        # # img = img.unsqueeze(0)
        
        # size_cut = [224,224,1];
        # mask = load_data.read_nii_position(self.mask_list[index], size_cut, transl)
        # mask[mask<0]=0
        # mask[mask>0]=255
        # mask = torch.tensor(mask.astype(np.bool_))
        # # mask = img.unsqueeze(0)
        
        return 