#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2022 Artem Moskalev
# Copyright (c) 2022 Marc Finzi
# Paper: "LieGG: Studying Learned Lie Group Generators ", Artem Moskalev, Anna Sepliarskaia, Ivan Sosnovik, Arnold Smeulders, NeurIPS 2022
# GitHub: https://github.com/amoskalev/liegg

import os
import numpy as np
import torch
from skimage.filters import gaussian

#Synthetic dataset based on https://github.com/mfinzi/equivariant-MLP/blob/master/emlp/datasets.py
class O5Synthetic(object):
    def __init__(self, N=1024):
        super().__init__()
        d=5
        self.dim = 2*d
        self.X = np.random.randn(N,self.dim)
        ri = self.X.reshape(-1,2,5)
        r1,r2 = ri.transpose(1,0,2)
        self.Y = np.sin(np.sqrt((r1**2).sum(-1)))-.5*np.sqrt((r2**2).sum(-1))**3 + (r1*r2).sum(-1)/(np.sqrt((r1**2).sum(-1))*np.sqrt((r2**2).sum(-1)))
        self.Y = self.Y[...,None]
        # One has to be careful computing mean and std in a way so that standardizing
        # does not violate equivariance
        Xmean = self.X.mean(0) # can add and subtract arbitrary tensors
        Xscale = (np.sqrt((self.X.reshape(N,2,d)**2).mean((0,2)))[:,None]+0*ri[0]).reshape(self.dim)
        self.stats = 0,Xscale,self.Y.mean(axis=0),self.Y.std(axis=0)

    def __getitem__(self,i):
        return (self.X[i],self.Y[i])
    
    def __len__(self):
        return self.X.shape[0]

class RotoMNIST(torch.utils.data.Dataset):
    def __init__(self, data_path, split, normalize=True, blur=None):
                
        self.split = split
        self.bpath = os.path.join(data_path, split)
        
        self.samples = np.sort(os.listdir(self.bpath))
        self.blur = blur
        
        #normalization
        self.normalize = normalize
        self.mean = 0.1303
        self.std = 0.3008
        
        #vectorized
        self.vec_x = None
        self.vec_y = None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        load_path = os.path.join(self.bpath, self.samples[idx])
        img, label = np.load(load_path, allow_pickle=True)
        
        if not self.blur is None:
            img = gaussian(img, sigma=self.blur)
        
        if self.normalize:
            img -= self.mean
            img /= self.std
        
        return img, label
    
    def get_vectorized(self):
        
        if self.vec_x is None or self.vec_y is None:
            vecX, vecY = [], []
            for i in range(len(self.samples)):
                img, label = self.__getitem__(i)
                vecX += [img[None]]
                vecY += [label]
                
            self.vec_x = np.vstack(vecX)
            self.vec_y = np.array(vecY)
        
        return self.vec_x, self.vec_y
