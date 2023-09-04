# -*- coding: utf-8 -*-
# Implementation of Select2Col.
# Author: Qian Huang <huangq@zhejianglab.com>, Yuntao Liu <liuyt@zhejianglab.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention
from opencood.models.sub_modules.gnn_layers import GraphConvolution
import os
import shutil

class SparseMapGenerator(nn.Module):
    def __init__(self, args):
        super(SparseMapGenerator, self).__init__()
        # Threshold of objectiveness
        self.threshold = args['threshold']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        else:
            self.smooth = False

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))

        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, B):
        """
        Args:
            batch_confidence_maps: [(L1, H, W), (L2, H, W), ...]
        """

        _, _, H, W = batch_confidence_maps[0].shape
        sparse_masks = []
        sparse_rates = []
        for b in range(B):
            ori_sparse_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)
            if self.smooth:
                sparse_maps = self.gaussian_filter(ori_sparse_maps)
            else:
                sparse_maps = ori_sparse_maps

            L = sparse_maps.shape[0]
            if self.training:
                # Official training proxy objective
                K = int(H * W * random.uniform(0, 1))
                sparse_maps = sparse_maps.reshape(L, H * W)
                _, indices = torch.topk(sparse_maps, k=K, sorted=False)
                sparse_mask = torch.zeros_like(sparse_maps).to(sparse_maps.device)
                ones_fill = torch.ones(L, K, dtype=sparse_maps.dtype, device=sparse_maps.device)
                sparse_mask = torch.scatter(sparse_mask, -1, indices, ones_fill).reshape(L, 1, H, W)
            elif self.threshold:
                ones_mask = torch.ones_like(sparse_maps).to(sparse_maps.device)
                zeros_mask = torch.zeros_like(sparse_maps).to(sparse_maps.device)
                sparse_mask = torch.where(sparse_maps > self.threshold, ones_mask, zeros_mask)
            else:
                sparse_mask = torch.ones_like(sparse_maps).to(sparse_maps.device)

            sparse_rate = sparse_mask.sum() / (L * H * W)
            # Ego
            sparse_mask[0] = 1

            sparse_masks.append(sparse_mask)
            sparse_rates.append(sparse_rate)
        sparse_rates = sum(sparse_rates) / B
        # print('self.training,sparse_rates:',self.training,sparse_rates)
        sparse_masks = torch.cat(sparse_masks, dim=0)
        return sparse_masks, sparse_rates

class CollaboratorSelection(nn.Module):
    def __init__(self, nin, nout):
        super(CollaboratorSelection, self).__init__()

        self.gcn = GraphConvolution(nin, nout)
        self.tanhAug = nn.Tanh()

    def forward(self, x, historical_x, sparse_maps, truely_time_delay_t, adj):
        sparse_maps_latency = torch.mul(torch.mean(torch.mul(sparse_maps,sparse_maps[0].unsqueeze(0)),dim=1).unsqueeze(1),truely_time_delay_t)
        enhance_weight = self.gcn(sparse_maps_latency, adj)
        enhance_weight = self.tanhAug(enhance_weight) + torch.Tensor([1.0]).cuda()
        x_enw = torch.zeros_like(x)  ## semantic information enhance weight
        historical_x_enw = torch.zeros_like(historical_x)  ## historical semantic information enhance weight
        for t in range(x.shape[0] + historical_x.shape[0]):
            if t == 1 or t == 2:
                historical_x_enw[t - 1] = enhance_weight[t] if enhance_weight[t]>0 else torch.tensor([0.0]).cuda()
            elif t >= 3:
                x_enw[t - 2] = enhance_weight[t] if enhance_weight[t]>0 else torch.tensor([0.0]).cuda()
            else:
                x_enw[t] = enhance_weight[t] if enhance_weight[t]>0 else torch.tensor([0.0]).cuda()
        x = x * x_enw
        historical_x = historical_x * historical_x_enw
        return x, historical_x

class TransformerFusion(nn.Module):
    def __init__(self, feature_dim):
        super(TransformerFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C), perform self attention on each pixel
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x

class ShortTermAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ShortTermAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class HPHA(nn.Module):
    def __init__(self, args):
        super(HPHA, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
        self.downsample_rate = args['downsample_rate']

        self.fully = args['fully']
        if self.fully:
            print('constructing a fully connected sparse graph')
        else:
            print('constructing a partially connected sparse graph')

        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            # layer_nums,self.num_levels,num_filters: [3, 5, 8] 3 [64, 128, 256]
            for idx in range(self.num_levels):
                fuse_network = TransformerFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
        else:
            self.fuse_modules = TransformerFusion(args['in_channels'])
        self.sparse_map_generator = SparseMapGenerator(args['sparse'])
        self.sta = ShortTermAttention(num_filters[-1]*2)
        self.collaborator_selection = CollaboratorSelection(nin=1, nout=1)
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, historical_x, psm_single, record_len, pairwise_t_matrix, orin_time_delay, backbone=None):
        """
        Fusion forwarding.

        Parameters:
            x: Input data, (sum(n_cav), C, H, W).
            record_len: List, (B).
            pairwise_t_matrix: The transformation matrix from each cav to ego, (B, L, L, 4, 4).

        Returns:
            Fused feature.
        """

        _, C, H, W = x.shape
        B = pairwise_t_matrix.shape[0]

        ## Collaborator Selection and Semantic Information Enhanced based on IoSI
        original_sparse_maps = self.regroup(psm_single,record_len)
        sparse_maps, sparse_rates = self.sparse_map_generator(original_sparse_maps, B)
        cav_num, C, H, W = sparse_maps.shape
        sparse_maps = sparse_maps.view(cav_num, -1) # [4, 8448]
        truely_time_delay = orin_time_delay[0][0:sparse_maps.shape[0]]
        truely_time_delay_full = torch.full(truely_time_delay.shape, 0.1).to(truely_time_delay.device)
        truely_time_delay_t = torch.tensor((torch.tensor(1.0).to(truely_time_delay.device) / (truely_time_delay + truely_time_delay_full)).unsqueeze(1),dtype=torch.float32) #truely_time_delay.shape [4,1]
        adj = torch.eye(truely_time_delay_t.shape[0]).to(truely_time_delay_t.device)
        x, historical_x = self.collaborator_selection(x, historical_x, sparse_maps, truely_time_delay_t, adj)
        historical_x = backbone.blocks[0](historical_x)

        ## Semantic Information Aggregated from Spatial Dimension based on Multi-scale Transformer Module
        ups = []
        for i in range(self.num_levels):
            x = backbone.blocks[i](x)
            batch_node_features = self.regroup(x, record_len)

            # Fusion
            x_fuse = []
            for b in range(B):
                neighbor_feature = batch_node_features[b]
                x_fuse.append(self.fuse_modules[i](neighbor_feature))
            x_fuse = torch.stack(x_fuse)
            if len(backbone.deblocks) > 0:
                ups.append(backbone.deblocks[i](x_fuse))
            else:
                ups.append(x_fuse)
        ## Semantic Information Refined from Temporal Dimension based on Short-term Attention Module
        ups.append(historical_x[0].unsqueeze(0))
        ups.append(historical_x[1].unsqueeze(0))
        print (len(ups))
        if len(ups) > 1:
            x_fuse = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x_fuse = ups[0]
        x_fuse = self.sta(x_fuse) * x_fuse
        if len(backbone.deblocks) > self.num_levels:
            x_fuse = backbone.deblocks[-1](x_fuse)
        return x_fuse
