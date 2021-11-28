from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.lib.arraysetops import unique
import torch.utils.data as torch_data

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder
from .processor.cylind_feat import CylinderFeatureEncoder

def cart2polar(xyz):
    rho = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.stack((rho, phi, xyz[:, 2]), axis=1)

def polar2cat(xyz_polar):
    x = xyz_polar[0] * np.cos(xyz_polar[1])
    y = xyz_polar[0] * np.sin(xyz_polar[1])
    return np.stack((x, y, xyz_polar[2]), axis=0)

def get_distance(x, y):
    return np.sqrt(x ** 2 + y ** 2)

class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.cylind_range = np.array(self.dataset_cfg.CYLIND_RANGE, dtype=np.float32) if hasattr(self.dataset_cfg, 'CYLIND_RANGE') else None
        self.cylind_size = np.array(self.dataset_cfg.CYLIND_SIZE, dtype=np.float32) if hasattr(self.dataset_cfg, 'CYLIND_SIZE') else None
        self.cylind_feats = self.dataset_cfg.CYLIND_FEATS if hasattr(self.dataset_cfg, 'CYLIND_FEATS') else False
        self.cart_feats = self.dataset_cfg.CART_FEATS if hasattr(self.dataset_cfg, 'CART_FEATS') else False
        self.cy_grid_size = np.round((self.cylind_range[3:6] - self.cylind_range[0:3]) / np.array(self.cylind_size)).astype(np.int64) if hasattr(self.dataset_cfg, 'CYLIND_RANGE') else None
        
        self.voxel_centers = self.dataset_cfg.VOXEL_CENTERS if hasattr(self.dataset_cfg, 'VOXEL_CENTERS') else False

        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )

        self.num_point_features = self.point_feature_encoder.num_point_features
        # self.num_point_features += 2
        # self.cylind_fea_model = CylinderFeatureEncoder(fea_dim=4, fea_compre=16)

        if self.cylind_feats and self.cart_feats:
            self.num_point_features = 5
        if self.voxel_centers:
            self.num_point_features += 3

        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        self.sum_grid_ind = np.zeros((0, 3))

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError
    
    def statistics(grid_ind, grid_size, nonempty_voxel_num):
        all_voxels = grid_size[0] / 8 * grid_size[1] * grid_size[2]
        for i in range(8):
            distance = (i + 1) * grid_size[0] / 8
            nonempty_voxel_num[i] += grid_ind[grid_ind[:, 0] < distance].shape
        

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
            if len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        if self.cylind_size is not None:
            # replace 'voxels'(V, max_num, C=4) and 'voxel_coords'(V, C=3) (L, W, H)  in data_dicts
            xyz = data_dict['points'][:, :3]
            intensity = data_dict['points'][:, 3][:, np.newaxis]

            xyz_pol = cart2polar(xyz)   # (N, 3)

            z_feats = xyz[:, 2:3]

            if self.cart_feats and self.cylind_feats:
                pol_feats = np.concatenate((xyz_pol[:, :2], xyz[:, :2], z_feats, intensity), axis=1)
            elif self.cart_feats:
                pol_feats = np.concatenate((xyz[:, :2], z_feats, intensity), axis=1)
            elif self.cylind_feats:
                pol_feats = np.concatenate((xyz_pol[:, :2], z_feats, intensity), axis=1)

            max_bound = np.array(self.cylind_range[3:6])
            min_bound = np.array(self.cylind_range[0:3])
            max_bound_1e4 = np.round(max_bound * 1e4).astype(np.int64)
            min_bound_1e4 = np.round(min_bound * 1e4).astype(np.int64)

            crop_range = self.cylind_range[3:6] - self.cylind_range[0:3]
            grid_size = crop_range / np.array(self.cylind_size)
            self.grid_size = np.round(grid_size).astype(np.int64)

            intervals = self.cylind_size
            intervals_1e4 = (self.cylind_size * 1e4).astype(np.int)

            if (intervals_1e4 == 0).any(): print("Zero interval!")
            
            remove_index = np.concatenate((np.where(xyz_pol * 1e4 < min_bound_1e4 + 1e-20)[0], np.where(xyz_pol * 1e4 > max_bound_1e4 - 1e-20)[0]))
            xyz_pol = np.delete(xyz_pol, remove_index, axis=0)
            pol_feats = np.delete(pol_feats, remove_index, axis=0)

            cy_grid_ind = (np.floor((np.clip(xyz_pol.astype(np.float64) * 1e4, min_bound_1e4, max_bound_1e4) - min_bound_1e4) / intervals_1e4)).astype(np.int)

            # unq, cy_feats = self.cylind_fea_model(pol_feats, cy_grid_ind)
            # sort potential repeated grid_inds first by the 1st col, then 3nd col, then 3rd col. 
            sorted_indices = np.lexsort((cy_grid_ind[:, 2], cy_grid_ind[:, 1], cy_grid_ind[:, 0]))
            sorted_pol_feats = pol_feats[sorted_indices]
            sorted_cy_grid_ind = cy_grid_ind[sorted_indices]

            unique_grid_ind, first_indexes, grid_cnts = np.unique(sorted_cy_grid_ind, axis=0, return_index=True, return_counts=True)
            
            if self.voxel_centers:
                voxel_centers = (unique_grid_ind.astype(np.float32) + 0.5) * intervals  + min_bound
                data_dict['voxel_centers'] = voxel_centers

            # get a list of all indices of unique elements in a numpy array
            sector_feats = np.split(sorted_pol_feats, first_indexes[1:])
            voxel_max_num = 5 #data_dict['voxels'].shape[1]
            sectors = np.zeros((unique_grid_ind.shape[0], voxel_max_num, (sector_feats[0].shape[1])))

            for i in range(len(sector_feats)):
                if sector_feats[i].shape[0] > 5:
                    grid_cnts[i] = 5
                    sectors[i, :, :sector_feats[i].shape[1]] = sector_feats[i][np.random.choice(sector_feats[i].shape[0], 5, replace=False)]
                    #sectors[i, :, sector_feats[i].shape[1]:] = np.expand_dims(unique_grid_ind[i, [1,0]], 0).repeat(5, 0)
                else:
                    point_num_in_sector = sector_feats[i].shape[0]
                    sectors[i, :point_num_in_sector, :sector_feats[i].shape[1]] = sector_feats[i]
                    #sectors[i, :point_num_in_sector, sector_feats[i].shape[1]:] = np.expand_dims(unique_grid_ind[i, [1,0]], 0).repeat(point_num_in_sector, 0)

            data_dict['voxel_coords'] = unique_grid_ind[:, [2, 1, 0]]
            data_dict['voxels'] = sectors
            data_dict['voxel_num_points'] = grid_cnts
            data_dict['pt_fea'] = pol_feats
            data_dict['xy_ind'] = cy_grid_ind


        # self.sum_grid_ind = np.unique(np.concatenate((self.sum_grid_ind, unique_grid_ind)), axis=0)
        # input_sp_tensor = spconv.SparseConvTensor(
        #     features=voxel_features,
        #     indices=voxel_coords.int(),
        #     spatial_shape=self.sparse_shape,
        #     batch_size=batch_size
        # )

        # voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        # dim_array = np.ones(len(self.grid_size) + 1, int)
        # dim_array[0] = -1
        # # voxel_position (3, H, W, L)
        # voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)  
        
        # import pdb
        # pdb.set_trace()

        # # get xyz of different polar: voxel_position: (3, H, W, Z)
        # voxel_position = polar2cat(voxel_position)

        # processed_label = np.ones(self.grid_size, dtype=np.uint8) #* self.ignore_label
        # #label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        # #label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        # #processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        
        # # processed_label: (H, W, Z)
        # data_tuple = (voxel_position, processed_label)

        # # center data on each voxel for PTnet
        # voxel_centers = (cy_grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        # return_xyz = xyz_pol - voxel_centers    # (N, 3) points' cylinder offset to voxel centers
        # return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)  # (N, 8)

        # return_fea = return_xyz

        # if self.return_test:
        #     data_tuple += (grid_ind, return_fea, index)
        # else:
        #     data_tuple += (grid_ind, return_fea)

        data_dict.pop('gt_names', None)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points', 'voxel_centers']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['pt_fea', 'xy_ind']:
                    ret[key] = val
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size

        return ret
