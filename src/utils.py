#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2022 Artem Moskalev
# Paper: "LieGG: Studying Learned Lie Group Generators ", Artem Moskalev, Anna Sepliarskaia, Ivan Sosnovik, Arnold Smeulders, NeurIPS 2022
# GitHub: https://github.com/amoskalev/liegg

import torch
import torch.nn as nn
import numpy as np

#################################
# data utils
#################################

def split_data(base_dataset, splits, seed=2022):
    
    # process splits
    split_values = np.array(list(splits.values()))
    assert (split_values == -1).sum() <= 1, "dict(splits) permits only one dynamyc argument"
    
    off_len = len(base_dataset) - split_values[split_values != -1].sum()
    split_values[split_values == -1] = off_len
    
    # random split
    splitted = torch.utils.data.random_split(base_dataset, 
                                             split_values,
                                             generator=torch.Generator().manual_seed(seed))
    # record to dict
    out_data = {}
    for i, eack_k in enumerate(splits.keys()):
        out_data[eack_k] = splitted[i]
        
    return out_data

#################################
# network utils
#################################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class L2_normed_net(nn.Module):
    def __init__(self, net):
        super().__init__()
        
        self.net = net

    def forward(self, x):
        
        xx = self.net(x)
                
        return nn.functional.normalize(xx, dim=1)