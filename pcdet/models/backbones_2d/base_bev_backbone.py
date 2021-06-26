import numpy as np
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, stride=1, padding=1):
        super(ResBlock, self).__init__()

        self.conv1_3x1 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,1),
                        stride=stride, padding=(padding, 0), bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_filters, eps=1e-3, momentum=0.01)
        self.relu1_1 = nn.ReLU()

        self.conv1_1x3 = nn.Conv2d(out_filters, out_filters, kernel_size=(1,3),
                        stride=stride, padding=(0, padding), bias=False)
        self.bn1_2 = nn.BatchNorm2d(out_filters, eps=1e-3, momentum=0.01)
        self.relu1_2 = nn.ReLU()

        self.conv2_3x1 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,1),
                        stride=stride, padding=(padding, 0), bias=False)
        self.bn2_1 = nn.BatchNorm2d(out_filters, eps=1e-3, momentum=0.01)
        self.relu2_1 = nn.ReLU()

        self.conv2_1x3 = nn.Conv2d(out_filters, out_filters, kernel_size=(1,3),
                        stride=stride, padding=(0, padding), bias=False)
        self.bn2_2 = nn.BatchNorm2d(out_filters, eps=1e-3, momentum=0.01)
        self.relu2_2 = nn.ReLU()
              
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1_3x1(x)
        shortcut = self.bn1_1(shortcut)
        shortcut = self.relu1_1(shortcut)

        shortcut = self.conv1_1x3(shortcut)
        shortcut = self.bn1_2(shortcut)
        shortcut = self.relu1_2(shortcut)

        res = self.conv2_3x1(x)
        res = self.bn2_1(res)
        res = self.relu2_1(res)

        res = self.conv2_1x3(res)
        res = self.bn2_2(res)
        res = self.relu2_2(res)

        res = res + shortcut

        return res


class AsymmBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, cylind_range=None, cy_grid_size=None):
        super().__init__()
        self.model_cfg = model_cfg
        
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        self.cylind = True if cy_grid_size is not None else False
        self.cy_grid_size = cy_grid_size
        self.cylind_range = cylind_range

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                ResBlock(c_in_list[idx], num_filters[idx], stride=layer_strides[idx], padding=0)
            ]

            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    ResBlock(num_filters[idx], num_filters[idx], padding=1)
                ])

            self.blocks.append(nn.Sequential(*cur_layers))
             
        if len(upsample_strides) > 0:
            stride = upsample_strides[idx]
            if stride >= 1:
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(
                        num_filters[idx], num_upsample_filters[idx],
                        upsample_strides[idx],
                        stride=upsample_strides[idx], bias=False
                    ),
                    nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ))
            else:
                stride = np.round(1 / stride).astype(np.int)
                self.deblocks.append(nn.Sequential(
                    nn.Conv2d(
                        num_filters[idx], num_upsample_filters[idx],
                        stride,
                        stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, cylind_range=None, cy_grid_size=None):
        super().__init__()
        self.model_cfg = model_cfg
        
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]

            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))

        if len(upsample_strides) > 0:
            stride = upsample_strides[idx]
            if stride >= 1:
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(
                        num_filters[idx], num_upsample_filters[idx],
                        upsample_strides[idx],
                        stride=upsample_strides[idx], bias=False
                    ),
                    nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ))
            else:
                stride = np.round(1 / stride).astype(np.int)
                self.deblocks.append(nn.Sequential(
                    nn.Conv2d(
                        num_filters[idx], num_upsample_filters[idx],
                        stride,
                        stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict