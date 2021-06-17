import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch
from torch.utils import data
import os, sys
sys.path.append(os.path.abspath('.'))

from pcdet.config import cfg, cfg_from_yaml_file
# from pcdet.datasets import DatasetTemplate
# from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, calibration_kitti 
from visual_utils import visualize_utils as V
from collections import defaultdict
import pickle
from skimage import io

# colormap: green[0,1,0] for 1, cyan[0,1,1] for 2, [1,1,0] for 3
label2idx = {'Car': 1, 'Cyclist': 2, 'Pedestrian': 3}
# classnames: ['Car', 'Pedestrian', 'Cyclist']

class DatasetTemplate(torch.utils.data.Dataset):
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
        # self.point_feature_encoder = PointFeatureEncoder(
        #     self.dataset_cfg.POINT_FEATURE_ENCODING,
        #     point_cloud_range=self.point_cloud_range
        # )
        # self.data_augmentor = DataAugmentor(
        #     self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        # ) if self.training else None
        # self.data_processor = DataProcessor(
        #     self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training
        # )

        # self.grid_size = self.data_processor.grid_size
        # self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

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

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        # data_dict = self.point_feature_encoder.forward(data_dict)

        # data_dict = self.data_processor.forward(
        #     data_dict=data_dict
        # )

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

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
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
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


import copy
def boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib):
    """
    Args:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        calib:

    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """
    boxes3d_camera_copy = copy.deepcopy(boxes3d_camera)
    xyz_camera, r = boxes3d_camera_copy[:, 0:3], boxes3d_camera_copy[:, 6:7]
    l, h, w = boxes3d_camera_copy[:, 3:4], boxes3d_camera_copy[:, 4:5], boxes3d_camera_copy[:, 5:6]

    xyz_lidar = calib.rect_to_lidar(xyz_camera)
    xyz_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, save_path=None, root_path=None, logger=None, ext='.bin', label_path=None, pred_path=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.vis_path = Path(save_path) if save_path else Path('data/kitti/training/result_vis')
        self.vis_path.mkdir(parents=True, exist_ok=True) 
        self.label_path = label_path
        self.pred_path = pred_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

        self.gt_file_list = glob.glob(str(label_path / f'*.txt')) if self.label_path.is_dir() else [self.label_path]
        self.gt_file_list.sort()

        with open(self.pred_path, 'rb') as f:
            self.pred_dict = pickle.load(f)

    def get_calib(self, idx):
        calib_file = self.label_path / '..' / 'calib' / ('%06d.txt' % idx)
        print('calib_file: ', calib_file)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_image_shape(self, idx):
        img_file = self.root_path / '..' /  'image_2' / ('%06d.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def __len__(self):
        return len(self.pred_dict)

    def __getitem__(self, index):
        pred_dict = self.pred_dict[index]
        frame_id = pred_dict['frame_id']

        print('frame_id: ', frame_id)
        index = int(frame_id[:6])
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        calib = self.get_calib(index)
        if self.dataset_cfg.FOV_POINTS_ONLY:
            img_shape = self.get_image_shape(index)
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]


        # dict_keys: 'name', 'truncated', 'occluded', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y', 'score', 'boxes_lidar', 'frame_id'
        # only use 'frame_id', 'name', 'bbox', 'alpha', 'score', 'bboxes_lidar'

        pred_labels = pred_dict['name']
        pred_scores = pred_dict['score']
        pred_boxes = pred_dict['boxes_lidar']

        gt_boxes = []
        gt_names = []
        with open(self.gt_file_list[index]) as gt_file:
            gt_data_lines = gt_file.readlines()
            for line in gt_data_lines:
                line = line.strip().split(' ')
                gt_data_line = line
                gt_boxes.append([float(num) for num in gt_data_line[-7:]])
                gt_names.append(line[0])

        gt_boxes = np.array(gt_boxes) 
        gt_names = np.array(gt_names)

        loc, rots = gt_boxes[:, 3:6], gt_boxes[:, -1]
        w, h, l = gt_boxes[:, 0:1], gt_boxes[:, 1:2], gt_boxes[:, 2:3]
        gt_boxes_camera = np.concatenate([loc, l, w, h, rots[..., np.newaxis]], axis=1).astype(np.float32)
        gt_boxes_lidar = boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
        
        input_dict = {
            'points': points,
            'frame_id': index,
            'pred_labels': pred_labels,
            'pred_scores': pred_scores,
            'pred_boxes': pred_boxes,
            'gt_boxes': gt_boxes_lidar,
            'gt_names': gt_names
        }

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='data/kitti/training/velodyne/',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--label_path', type=str, help='specify the directory of the ground truth file')
    
    parser.add_argument('--pred_path', type=str, default='result.pkl', help='specify the directory of the prediction results (one total evaluation in one pickle file)')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    print(args.pred_path) 
    assert os.path.exists(args.pred_path)
    
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger,
        label_path=Path(args.label_path), pred_path=Path(args.pred_path),
        save_path=Path(args.save_path)
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    # model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    # model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    # model.cuda()
    # model.eval()
    # with torch.no_grad():
    #     for idx, data_dict in enumerate(demo_dataset):
    #         logger.info(f'Visualized sample index: \t{idx + 1}')
    #         data_dict = demo_dataset.collate_batch([data_dict])
    #         load_data_to_gpu(data_dict)
    #         pred_dicts, _ = model.forward(data_dict)

    #         V.draw_scenes(
    #             points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
    #             ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
    #         )
    #         mlab.show(stop=True)

    for idx, data_dict in enumerate(demo_dataset):
        data_dict = demo_dataset.collate_batch([data_dict])
        ref_labels = np.array([label2idx[label] for label in data_dict['pred_labels'][0]])
        # params of draw_scenes:
        #   ref_boxes: (N, 7), float
        #   ref_scores: (N), float
        #   ref_labels: (N), int
        #   gt_boxes: (N, 8), float
        frame_id = data_dict['frame_id'].item()
        V.draw_scenes(
            points=data_dict['points'][:, 1:], 
            gt_boxes=data_dict['gt_boxes'][0],
            ref_boxes=data_dict['pred_boxes'][0],
            ref_scores=data_dict['pred_scores'][0], 
            ref_labels=ref_labels,
            frame_id=frame_id
        )
        # mlab.show(stop=True)
        print('root_path: ', demo_dataset.root_path)
        mlab.savefig(str(demo_dataset.vis_path / ('%06d.png' % frame_id)))
        print('save_path: ', str(demo_dataset.vis_path / ('%06d.png' % frame_id)))

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
