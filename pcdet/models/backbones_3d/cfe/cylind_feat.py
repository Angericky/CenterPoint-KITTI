# -*- coding:utf-8 -*-
# author: Xinge

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb
import multiprocessing
import torch_scatter


class CylinderFeatureEncoder(nn.Module):
    def __init__(self, model_cfg):
        super(CylinderFeatureEncoder, self).__init__()

        fea_dim = model_cfg.get('FEA_DIM', 3)
        out_pt_fea_dim = model_cfg.get('OUT_PT_FEA_DIM', 64)
        max_pt_per_encode = model_cfg.get('MAT_PT_PER_ENCODE',64)
        fea_compre = model_cfg.get('FEA_COMPRE',None)
        
        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),

            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, out_pt_fea_dim)
        )

        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)
        self.pool_dim = out_pt_fea_dim

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, batch_dict):
        pt_fea, xy_ind = batch_dict['pt_fea'], batch_dict['xy_ind']

        cur_dev = pt_fea[0].get_device()

        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            lwh_ind = xy_ind[i_batch]
            hwl_ind = lwh_ind[:, [2, 1, 0]]
            cat_pt_ind.append(F.pad(hwl_ind, (1, 0), 'constant', value=i_batch))

        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]
        
        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        # process feature
        processed_cat_pt_fea = self.PPmodel(cat_pt_fea)
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]

        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data
        
        batch_dict['voxel_features'] = processed_pooled_data
        batch_dict['voxel_coords'] = unq

        return batch_dict