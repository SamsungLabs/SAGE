import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange
import wandb

import numpy as np
import os
import random
import ipdb
import time

from mixup import to_one_hot

def normalize_grad(grad, alpha=1):

    grad.div_(grad.sum(dim=[1,2,3],keepdim=True))
    grad.mul_(np.clip(alpha, a_min=1e-12, a_max=None))

    return grad

def return_center(padded_input,w):
    return padded_input[:,:,w:2*w,w:2*w]

def pad_zeros(input_2b_padded, left, right, top, bottom):
    padded_input = torch.nn.functional.pad(input_2b_padded, [left, right, top, bottom], mode='constant', value=0.0)
    return padded_input

def reweighted_lam(mixed_y, mixed_lam, num_classes):
    y0 = to_one_hot(mixed_y[0], num_classes)
    y1 = to_one_hot(mixed_y[1], num_classes)

    return mixed_lam[0].unsqueeze(1)*y0 + mixed_lam[1].unsqueeze(1)*y1

def sage(x, y, grad, alpha=1, rand_pos=1):
    '''Returns mixed inputs, pairs of targets, and lambda'''
####################################################################################
    # init a bunch of variables
    batch_size, c, w, h = np.array(x.size())
    index = torch.randperm(batch_size, device = x.device)

    mixed_y = [y, y[index]]
    padded_grad_1 = pad_zeros(grad, w,w,w,w)
    padded_grad_2 = padded_grad_1[index, :]

    best_ij = torch.zeros([batch_size, 2], dtype=int, device = x.device)
    mixed_x = torch.zeros_like(x, device = x.device)

    padded_normalized_grad_1 = normalize_grad(padded_grad_1, alpha)
    padded_normalized_grad_2 = normalize_grad(padded_grad_2, 1-alpha)
    padded_x_1 = pad_zeros(x, w,w,w,w)
    padded_x_2 = padded_x_1[index,:]

    possible_positions_per_axis = int((2*w-1))
    actual_positions = int(possible_positions_per_axis**2*rand_pos)
    _x = torch.linspace(1, 2*w-1, possible_positions_per_axis, dtype=int, device=x.device)    
    _xv, _yv = torch.meshgrid(_x, _x, indexing='xy')
    rand_perm = torch.randperm(possible_positions_per_axis**2, device=x.device)
    # comment/uncomment here for reproducing the same results
    coord = torch.stack((_xv.flatten(), _yv.flatten()))[:,rand_perm[:actual_positions]]
#     coord = torch.stack((_xv.flatten(), _yv.flatten()))[:,:actual_positions]
    
#     coord[:,0],coord[:,1] = w,w # added July 25 after 1st round of tiny imagenet results:
####################################################################################
    # iterate over images and find the best position that maximizes saliency
    theta = torch.eye(2,3,device=x.device).repeat(coord.shape[1],1,1)
    theta[:,0,2] = 2*(w-coord[1,:])/(3*w)
    theta[:,1,2] = 2*(w-coord[0,:])/(3*w)
    size = torch.Size((coord.shape[1],1,3*w,3*w))
    grid = F.affine_grid(theta, size, align_corners=False)
    
    for img in range(batch_size):
        single_padded_normalized_grad_2 = padded_normalized_grad_2[img].expand(coord.shape[1],1,3*w,3*w)
        translated_single_padded_normalized_grad_2 = F.grid_sample(single_padded_normalized_grad_2,
                                                             grid,
                                                             mode='nearest',
                                                             padding_mode ='zeros',
                                                             align_corners=False)

        single_padded_normalized_grad_1 = padded_normalized_grad_1[img].expand(coord.shape[1],1,3*w,3*w)

#         M =  single_padded_normalized_grad_1 / (single_padded_normalized_grad_1+translated_single_padded_normalized_grad_2+1e-12)
        M = single_padded_normalized_grad_1 / (single_padded_normalized_grad_1+translated_single_padded_normalized_grad_2).clamp(min=1e-16)

        saliency = return_center(single_padded_normalized_grad_1 * M+(translated_single_padded_normalized_grad_2 * (1-M)), w)

        best_ij[img,:] = coord[:,saliency.sum(dim=[1,2,3]).argmax()]

####################################################################################
    # update mixed images
    theta = torch.eye(2,3,device=x.device).repeat(batch_size,1,1)
    theta[:,0,2] = 2*(w-best_ij[:,1])/(3*w)
    theta[:,1,2] = 2*(w-best_ij[:,0])/(3*w)
    size = torch.Size((batch_size,c,3*w,3*w))
    grid = F.affine_grid(theta, size, align_corners=False)

    translated_padded_normalized_grad_2 = F.grid_sample(padded_normalized_grad_2,
                                                         grid,
                                                         mode='nearest',
                                                         padding_mode ='zeros',
                                                         align_corners=False)
    M = padded_normalized_grad_1 / (padded_normalized_grad_1+translated_padded_normalized_grad_2).clamp(min=1e-16)
    lambbda = return_center(M,w).mean(dim=[1,2,3])
    translated_padded_x_2 = F.grid_sample(padded_x_2,
                                          grid,
                                          mode='nearest',
                                          padding_mode ='zeros',
                                          align_corners=False)
    mixed_x = return_center(torch.mul(padded_x_1, M) + torch.mul(translated_padded_x_2, 1-M), w)
 
    mixed_lam = [lambbda.detach(), 1 - lambbda.detach()]

    del padded_grad_1, padded_grad_2, best_ij, padded_normalized_grad_1, padded_normalized_grad_2
    del padded_x_1, padded_x_2, possible_positions_per_axis, actual_positions, _x, _xv, _yv, coord
    del theta, size, grid, single_padded_normalized_grad_2, translated_single_padded_normalized_grad_2
    del M, saliency, lambbda, translated_padded_x_2

    return mixed_x.detach(), mixed_y, mixed_lam
