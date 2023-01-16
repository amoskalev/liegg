#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2022 Artem Moskalev
# Copyright (c) 2022 Marc Finzi
# Paper: "LieGG: Studying Learned Lie Group Generators ", Artem Moskalev, Anna Sepliarskaia, Ivan Sosnovik, Arnold Smeulders, NeurIPS 2022
# GitHub: https://github.com/amoskalev/liegg
# Parts of the code from: https://github.com/mfinzi/equivariant-MLP

import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self,x):
        return x.sigmoid()*x

def MLPBlock(cin,cout):
    return nn.Sequential(nn.Linear(cin, cout, bias=False), 
                         Swish())

class MLP(nn.Module):
    """ Standard baseline MLP. """
    def __init__(self, in_dim, out_dim, ch=384, num_nonlins=3):
        super().__init__()
        chs = [in_dim] + num_nonlins*[ch]
        cout = out_dim

        self.net = nn.Sequential(
            *[MLPBlock(cin,cout) for cin, cout in zip(chs,chs[1:])],
            nn.Linear(chs[-1], cout, bias=False)
        )

    def forward(self,x):
        y = self.net(x)
        return y

class Standardize(nn.Module):
    """ A convenience module to wrap a given module, normalize its input
        by some dataset x mean and std stats, and unnormalize its output by
        the dataset y mean and std stats. 
        Args:
            model (Module): model to wrap
            ds_stats ((μx,σx,μy,σy) or (μx,σx)): tuple of the normalization stats
        
        Returns:
            Module: Wrapped model with input normalization (and output unnormalization)"""
    def __init__(self,model,ds_stats):
        super().__init__()
        self.model = model
        self.ds_stats=ds_stats
        muin,sin,muout,sout = self.ds_stats
        self.muin = torch.tensor(muin)
        self.sin = torch.tensor(sin)
        self.muout = torch.tensor(muout)
        self.sout = torch.tensor(sout)

    def forward(self,x):
        if len(self.ds_stats)==2:
            muin,sin = self.ds_stats
            return self.model((x-muin)/sin)
        else:
            muin,sin,muout,sout = self.ds_stats
            y = self.sout*self.model((x - self.muin.float())/self.sin.float())+self.muout
            return y.float()
