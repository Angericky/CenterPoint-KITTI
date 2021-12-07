import numpy as np
import spconv
import torch
from torch import nn
from functools import partial

# features: (Z, phi, rho)
def conv3x3(in_planes, out_planes, stride=1, dilation=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, dilation=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride, dilation=dilation,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)

def conv3x1(in_planes, out_planes, stride=1, dilation=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 3, 1), stride=stride, dilation=dilation,
                             padding=(1, 1, 0), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, dilation=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride, dilation=dilation,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, dilation=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride, dilation=dilation,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, dilation=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride, dilation=dilation, 
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, dilation=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride, dilation=dilation,
                             padding=1, bias=False, indice_key=indice_key)


def subm_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0):
    norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
    conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.LeakyReLU(inplace=True),
    )

    return m

class ResBlock(spconv.SparseModule):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=2, padding=1, dilation=1, 
                 pooling=True, indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling

        self.conv1 = conv3x1(in_filters, out_filters, dilation=dilation, indice_key=indice_key + "bef")
        self.act1 = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(out_filters)

        # self.conv1_block = spconv.SparseSequential(
        #     conv1,
        #     norm_fn(out_channels),
        #     nn.LeakyReLU(inplace=True),
        # )

        self.conv1_2 = conv1x3(out_filters, out_filters, dilation=dilation, indice_key=indice_key + "bef")
        self.act1_2 = nn.LeakyReLU(inplace=True)
        self.bn1_2 = nn.BatchNorm1d(out_filters)

        # self.conv1_2_block = spconv.SparseSequential(
        #     conv1_2,
        #     norm_fn(out_channels),
        #     nn.LeakyReLU(inplace=True),
        # )

        self.conv2 = conv1x3(in_filters, out_filters, dilation=dilation, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv2_2 = conv3x1(out_filters, out_filters, dilation=dilation, indice_key=indice_key + "bef")
        self.act2_2 = nn.LeakyReLU(inplace=True)
        self.bn2_2 = nn.BatchNorm1d(out_filters)

        if pooling:
            self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=kernel_size, stride=stride,
                                                padding=padding, indice_key=indice_key, bias=False)

        self.weight_initialization()


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.features = self.bn1(shortcut.features)
        shortcut.features = self.act1(shortcut.features)

        shortcut = self.conv1_2(shortcut)
        shortcut.features = self.bn1_2(shortcut.features)
        shortcut.features = self.act1_2(shortcut.features)
        
        resA = self.conv2(x)
        resA.features = self.bn2(resA.features)
        resA.features = self.act2(resA.features)
    
        resA = self.conv2_2(resA)
        resA.features = self.bn2_2(resA.features)
        resA.features = self.act2_2(resA.features)

        resA.features = resA.features + shortcut.features

        if self.pooling:
            resB = self.pool(resA)
            return resB#, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU(inplace=True)
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x3(out_filters, out_filters, indice_key=indice_key)
        self.act3 = nn.LeakyReLU(inplace=True)
        self.bn3 = nn.BatchNorm1d(out_filters)
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA.features = self.trans_bn(upA.features)
        upA.features = self.trans_act(upA.features)
        
        ## upsample
        upA = self.up_subm(upA)

        upA.features = upA.features + skip.features

        upE = self.conv1(upA)
        upE.features = self.bn1(upE.features)
        upE.features = self.act1(upE.features)

        upE = self.conv2(upE)
        upE.features = self.bn2(upE.features)
        upE.features = self.act2(upE.features)

        upE = self.conv3(upE)
        upE.features = self.bn3(upE.features)
        upE.features = self.act3(upE.features)

        return upE


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.features = self.bn0(shortcut.features)
        shortcut.features = self.act1(shortcut.features)

        shortcut2 = self.conv1_2(x)
        shortcut2.features = self.bn0_2(shortcut2.features)
        shortcut2.features = self.act1_2(shortcut2.features)

        shortcut3 = self.conv1_3(x)
        shortcut3.features = self.bn0_3(shortcut3.features)
        shortcut3.features = self.act1_3(shortcut3.features)
        shortcut.features = shortcut.features + shortcut2.features + shortcut3.features

        shortcut.features = shortcut.features * x.features

        return shortcut


def subm_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0):
    norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
    conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(inplace=True),
    )

    return m


class dilate_block(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, indice_key=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block1 = ResBlock(in_channels, out_channels, padding=padding,indice_key='%s_1' % indice_key)
        self.block2 = ResBlock(out_channels, out_channels, padding=padding, dilation=2, pooling=False, indice_key='%s_2' % indice_key)
        self.block3 = ResBlock(out_channels, out_channels, padding=padding, dilation=3, pooling=False, indice_key='%s_3' % indice_key)
        self.downdim = conv1x1(out_channels * 3, out_channels, indice_key='%s_down' % indice_key)

    def forward(self, x):
        x_1 = self.block1(x)
        x_2 = self.block2(x_1)
        x_3 = self.block3(x_2)

        x_1.features = torch.cat((x_1.features, x_2.features, x_3.features), dim=-1)
        output = self.downdim(x_1)
        
        return output

class Asymm_3d_spconv(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, cy_grid_size=None, init_size=16, **kwargs):
        super().__init__()
        
        self.input_dim = model_cfg.get('CYLIND_FEAT', None)
        if self.input_dim == None:
            self.use_conv_input = True
        else:
            self.use_conv_input = False
        self.dilate = model_cfg.get('DILATE', None)

        if cy_grid_size is not None:
            
            sparse_shape = np.array(grid_size)   # shape for (H, W, L)
            # sparse_shape[0] = 11
            self.sparse_shape = cy_grid_size[::-1] + [1, 0, 0]

            if self.use_conv_input:
                self.conv_input = ResBlock(input_channels, init_size, pooling=False, indice_key="pre")
            # [1624, 1496, 41] <- [812, 748, 21]
            # print('dilate: ',self.dilate)
            # import pdb
            # pdb.set_trace()

            if self.dilate:
                # self.conv1 = spconv.SparseSequential(
                #     # subm_block(init_size, init_size, 3, padding=1, indice_key='subm1'),
                #     # subm_block(init_size, init_size, 3, padding=1, indice_key='subm1')
                #     #ResBlock(init_size, init_size, dilation=2, pooling=False, indice_key="subm1"),
                #     #ResBlock(init_size, init_size, dilation=3, pooling=False, indice_key="subm1"),
                # )
                # [1600, 1408, 41] <- [800, 704, 21]
                
                self.conv2 = dilate_block(init_size, 2 * init_size, indice_key="down2")
                    #ResBlock(2 * init_size, 2 * init_size, dilation=2, pooling=False, indice_key="subm2"),
                    #ResBlock(2 * init_size, 2 * init_size, dilation=3, pooling=False, indice_key="subm2"),
                    # subm_block(2 * init_size, 2 * init_size, 3, padding=1, indice_key='subm2'),
                    # subm_block(2 * init_size, 2 * init_size, 3, padding=1, indice_key='subm2'),
                
                # [812, 748, 21] <- [406, 374, 11]

                # [800, 704, 21] <- [400, 352, 11]
                
                self.conv3 = dilate_block(2 * init_size, 4 * init_size, indice_key="down3")
                    #ResBlock(4 * init_size, 4 * init_size, pooling=False, indice_key="subm3"),
                    #ResBlock(4 * init_size, 4 * init_size, pooling=False, indice_key="subm3"),
                    # subm_block(4 * init_size, 4 * init_size, 3, padding=1, indice_key='subm3'),
                    # subm_block(4 * init_size, 4 * init_size, 3, padding=1, indice_key='subm3'),
                
                # [406, 374, 11] <- [203, 187, 5]

                # [400, 352, 11] <- [200, 352, 5]s
                self.conv4 = dilate_block(4 * init_size, 8 * init_size, padding=(0, 1, 1), indice_key="down4")
                    #ResBlock(8 * init_size, 8 * init_size, pooling=False, indice_key="subm4"),
                    #ResBlock(8 * init_size, 8 * init_size, pooling=False, indice_key="subm4"),
                    # subm_block(8 * init_size, 8 * init_size, 3, padding=1, indice_key='subm4'),
                    # subm_block(8 * init_size, 8 * init_size, 3, padding=1, indice_key='subm4'),
                
            
                self.conv_output = ResBlock(8 * init_size, 8 * init_size, kernel_size=(3, 1, 1), padding=0, stride=(2, 1, 1), indice_key="out") 
    
            else:
                self.conv1 = spconv.SparseSequential(
                    # subm_block(init_size, init_size, 3, padding=1, indice_key='subm1'),
                    # subm_block(init_size, init_size, 3, padding=1, indice_key='subm1')
                    #ResBlock(init_size, init_size, pooling=False, indice_key="subm1"),
                    #ResBlock(init_size, init_size, pooling=False, indice_key="subm1"),
                )
                # [1600, 1408, 41] <- [800, 704, 21]
                
                self.conv2 = spconv.SparseSequential(
                    ResBlock(init_size, 2 * init_size, indice_key="down2"),
                    #ResBlock(2 * init_size, 2 * init_size, pooling=False, indice_key="subm2"),
                    #ResBlock(2 * init_size, 2 * init_size, pooling=False, indice_key="subm2"),
                    # subm_block(2 * init_size, 2 * init_size, 3, padding=1, indice_key='subm2'),
                    # subm_block(2 * init_size, 2 * init_size, 3, padding=1, indice_key='subm2'),
                )
                # [812, 748, 21] <- [406, 374, 11]

                # [800, 704, 21] <- [400, 352, 11]
                
                self.conv3 = spconv.SparseSequential(
                    ResBlock(2 * init_size, 4 * init_size, indice_key="down3"),
                    #ResBlock(4 * init_size, 4 * init_size, pooling=False, indice_key="subm3"),
                    #ResBlock(4 * init_size, 4 * init_size, pooling=False, indice_key="subm3"),
                    # subm_block(4 * init_size, 4 * init_size, 3, padding=1, indice_key='subm3'),
                    # subm_block(4 * init_size, 4 * init_size, 3, padding=1, indice_key='subm3'),
                )
                # [406, 374, 11] <- [203, 187, 5]

                # [400, 352, 11] <- [200, 352, 5]s
                self.conv4 = spconv.SparseSequential(
                    ResBlock(4 * init_size, 8 * init_size, padding=(0, 1, 1), indice_key="down4"),
                    # ResBlock(8 * init_size, 8 * init_size, pooling=False, indice_key="subm4"),
                    # ResBlock(8 * init_size, 8 * init_size, pooling=False, indice_key="subm4"),
                    # subm_block(8 * init_size, 8 * init_size, 3, padding=1, indice_key='subm4'),
                    # subm_block(8 * init_size, 8 * init_size, 3, padding=1, indice_key='subm4'),
                )
            
                self.conv_output = ResBlock(8 * init_size, 8 * init_size, kernel_size=(3, 1, 1), padding=0, stride=(2, 1, 1), indice_key="out") 
    
        else:
            sparse_shape = np.array(grid_size)   # shape for (H, W, L)
            # sparse_shape[0] = 11
            self.sparse_shape = grid_size[::-1] + [1, 0, 0]

            self.conv_input = ResContextBlock(input_channels, init_size, indice_key="pre")
            # [1600, 1408, 41] <- [800, 704, 21]
            self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
            # [800, 704, 21] <- [400, 352, 11]
            self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
            # [400, 352, 11] <- [200, 352, 5]
            self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, height_pooling=True, padding=(0, 1, 1),
                                    indice_key="down4")

            self.conv_output = ResBlock(8 * init_size, 8 * init_size, 0.2, height_pooling=False, kernel_size=(3, 1, 1), padding=0, indice_key="out")
        self.num_point_features = 128


    def forward(self, batch_dict):
        voxel_features, coors = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        x = spconv.SparseConvTensor(features=voxel_features, indices=coors, spatial_shape=self.sparse_shape,
                                      batch_size=batch_size)

        if self.use_conv_input:
            x = self.conv_input(x)

        if not self.dilate:
            x_conv1 = self.conv1(x)
        else:
            x_conv1 = x

        down1c = self.conv2(x_conv1)
        down2c = self.conv3(down1c)
        down3c = self.conv4(down2c)

        out = self.conv_output(down3c)

        # import pdb
        # pdb.set_trace()

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': down1c,
                'x_conv2': down2c,
                'x_conv3': down3c,
                'x_conv4': out,
            }
        })

        return batch_dict


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out
        

class Asymm_3d_res_spconv(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, cy_grid_size=None, init_size=16, **kwargs):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        self.input_dim = model_cfg.get('CYLIND_FEAT', None)
        if self.input_dim == None:
            self.use_conv_input = True
        else:
            self.use_conv_input = False

        if cy_grid_size is not None:
            sparse_shape = np.array(grid_size)   # shape for (H, W, L)
            # sparse_shape[0] = 11
            self.sparse_shape = cy_grid_size[::-1] + [1, 0, 0]

            if self.use_conv_input:
                self.conv_input = ResBlock(input_channels, init_size, pooling=False, indice_key="pre")
            # [1624, 1496, 41] <- [812, 748, 21]
            
            self.conv1 = spconv.SparseSequential(
                # subm_block(init_size, init_size, 3, padding=1, indice_key='subm1'),
                # subm_block(init_size, init_size, 3, padding=1, indice_key='subm1')
                SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
                SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
                # ResBlock(init_size, init_size, pooling=False, indice_key="subm1"),
                # ResBlock(init_size, init_size, pooling=False, indice_key="subm1"),
            )
            # [1600, 1408, 41] <- [800, 704, 21]
            
            self.conv2 = spconv.SparseSequential(
                ResBlock(init_size, 2 * init_size, indice_key="down2"),
                # ResBlock(2 * init_size, 2 * init_size, pooling=False, indice_key="subm2"),
                # ResBlock(2 * init_size, 2 * init_size, pooling=False, indice_key="subm2"),
                # subm_block(2 * init_size, 2 * init_size, 3, padding=1, indice_key='subm2'),
                # subm_block(2 * init_size, 2 * init_size, 3, padding=1, indice_key='subm2'),
                SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
                SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            )
            # [812, 748, 21] <- [406, 374, 11]

            # [800, 704, 21] <- [400, 352, 11]
            
            self.conv3 = spconv.SparseSequential(
                ResBlock(2 * init_size, 4 * init_size, indice_key="down3"),
                # ResBlock(4 * init_size, 4 * init_size, pooling=False, indice_key="subm3"),
                # ResBlock(4 * init_size, 4 * init_size, pooling=False, indice_key="subm3"),
                SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
                SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            )
            # [406, 374, 11] <- [203, 187, 5]

            # [400, 352, 11] <- [200, 352, 5]s
            self.conv4 = spconv.SparseSequential(
                ResBlock(4 * init_size, 8 * init_size, padding=(0, 1, 1), indice_key="down4"),
                # ResBlock(8 * init_size, 8 * init_size, pooling=False, indice_key="subm4"),
                # ResBlock(8 * init_size, 8 * init_size, pooling=False, indice_key="subm4"),
                # subm_block(8 * init_size, 8 * init_size, 3, padding=1, indice_key='subm4'),
                # subm_block(8 * init_size, 8 * init_size, 3, padding=1, indice_key='subm4'),
                SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
                SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            )
            
            self.conv_output = ResBlock(8 * init_size, 8 * init_size, kernel_size=(3, 1, 1), padding=0, stride=(2, 1, 1), indice_key="out") 
    
        else:
            sparse_shape = np.array(grid_size)   # shape for (H, W, L)
            # sparse_shape[0] = 11
            self.sparse_shape = grid_size[::-1] + [1, 0, 0]

            self.conv_input = ResContextBlock(input_channels, init_size, indice_key="pre")
            # [1600, 1408, 41] <- [800, 704, 21]
            self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
            # [800, 704, 21] <- [400, 352, 11]
            self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
            # [400, 352, 11] <- [200, 352, 5]
            self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, height_pooling=True, padding=(0, 1, 1),
                                    indice_key="down4")

            self.conv_output = ResBlock(8 * init_size, 8 * init_size, 0.2, height_pooling=False, kernel_size=(3, 1, 1), padding=0, indice_key="out")
        self.num_point_features = 128


    def forward(self, batch_dict):
        voxel_features, coors = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        x = spconv.SparseConvTensor(features=voxel_features, indices=coors, spatial_shape=self.sparse_shape,
                                      batch_size=batch_size)

        if self.use_conv_input:
            x = self.conv_input(x)

        x_conv1 = self.conv1(x)

        down1c = self.conv2(x_conv1)
        down2c = self.conv3(down1c)
        down3c = self.conv4(down2c)

        out = self.conv_output(down3c)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': down1c,
                'x_conv3': down2c,
                'x_conv4': down3c,
            }
        })

        return batch_dict
