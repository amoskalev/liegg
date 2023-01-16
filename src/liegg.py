#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2022 Artem Moskalev
# Paper: "LieGG: Studying Learned Lie Group Generators ", Artem Moskalev, Anna Sepliarskaia, Ivan Sosnovik, Arnold Smeulders, NeurIPS 2022
# GitHub: https://github.com/amoskalev/liegg

import torch
import torch.nn as nn
import numpy as np
import random

#################################
# polarization matrix
#################################

def polarization_matrix_1(model, data, dim = 5):
    # data: torch.FloatTensor(B, 2*dim)
    
    B = data.shape[0]

    data.requires_grad = True
    data.retain_grad()
    
    # compute network grads
    model.eval()
    y_pred = model(data)
    y_pred.backward(gradient=torch.ones_like(y_pred))
    
    # get grads and data per input dimension
    dF_1 = data.grad[...,:dim].view(B, dim, 1)
    data_1 = data[...,:dim].view(B, 1, dim)
    
    dF_2 = data.grad[...,dim:].view(B, dim, 1)
    data_2 = data[...,dim:].view(B, 1, dim)
    
    # collect into the network polarization matrix
    C = torch.bmm(dF_1, data_1) + torch.bmm(dF_2, data_2)
    
    return C.reshape(B, -1)

def polarization_matrix_2(model, data):
    # LieGG implementation with the groups acting on R^2
    # data: torch.FloatTensor(B, 28, 28)
    
    B, H, W = data.shape

    # compute image grads
    data_grad_x = data[:, 1:, :-1] - data[:, :-1, :-1]
    data_grad_y = data[:, :-1, 1:] - data[:, :-1, :-1]
    dI = torch.stack([data_grad_x, data_grad_y], -1)
    
    _,h,w,_ = dI.shape

    # compute network grads
    data = data.reshape(B, -1)
    data.requires_grad = True
    data.retain_grad()

    output = model(data)
    output.backward(torch.ones_like(output))

    dF = data.grad.reshape(B, H, W)
    dF = dF[:, :-1, :-1]
    
    # coordinate mask
    xy = torch.meshgrid(torch.arange(0, h), torch.arange(0, h))
    xy = torch.stack(xy, -1).to(dI.device)
    xy = xy / (H // 2) - 1
    
    # collect into the network polarization matrix
    C = dF[..., None, None] * dI[..., None] * xy[None, :, :, None, :]
    C = C.view(B, -1, 2, 2).sum(1)
    
    return C.reshape(B, -1)

#################################
# metrics
#################################

def closest_skew_sym(X):
    # X: NxN matrix 
    return 0.5*(X - X.transpose(-1,-2))

def symmetry_metrics(X):
    # X: torch.FloatTensor(B, d) <- Network polarization matrix

    N, d = X.shape
    root_d = int(d**0.5)

    _,S,Vh = torch.svd(X, compute_uv=True)

    sym_bias, generators = [], []
    for i in range(Vh.shape[0]):
        
        # save genertor
        generators += [Vh[:,i].reshape(root_d, root_d)]

        # compute symmetry_bias_i
        H_i = generators[-1]
        closest_H_i = closest_skew_sym(H_i)
        sym_bias_i = torch.norm(closest_H_i - H_i)
        sym_bias += [sym_bias_i]

    sym_bias = torch.FloatTensor(sym_bias)

    # compute symmetry variance
    sym_sings = (S**2)/N

    return sym_sings, sym_bias, generators