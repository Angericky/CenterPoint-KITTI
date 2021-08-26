import functools
import pdb
import torch.nn.functional as F
import torch
import math
import numpy as np
import torch.nn as nn
from torch.utils import data
from ...utils import box_coder_utils, common_utils, loss_utils, box_utils

from functools import partial
from six.moves import map, zip

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments
    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def convert_lidar_coords_to_image_coords(points, W, H, pc_range_min, resolution_w, resolution_h, color=[255, 255, 255]):
    bev = np.zeros((H, W, 3)) #
    
    x = (points[:, 0] - pc_range_min[0]) / resolution_w
    y = (points[:, 1] - pc_range_min[1]) / resolution_h
    x = x.astype(np.int)
    y = y.astype(np.int)
    out_mask = np.logical_or(np.logical_or(np.logical_or(x >= bev.shape[1], x < 0), y >= bev.shape[0]), y < 0)
    
    x_index = x[~out_mask]
    y_index = y[~out_mask]

    bev[y_index, x_index] = np.array(color)
    return bev


class CenterHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 cylind_range=None, cy_grid_size=None, predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = [class_names]
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)
        self.cylind = self.model_cfg.get('CYLIND_GRID', False)

        target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.target_cfg = target_cfg

        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range

        self.cy_grid_size = cy_grid_size
        self.cylind_range = cylind_range

        self.forward_ret_dict = {}
        
        #self.num_reg_channels = 7 if self.cylind else 8
        self.num_reg_channels = 8

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_reg_channels,
            kernel_size=1
        )

        self.loss_cls = GaussianFocalLoss(reduction='mean', 
            alpha=model_cfg.get('ALPHA', 2.0), gamma=model_cfg.get('GAMMA', 4.0))
        

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        # pay attention to preds if you use cylinderical partition
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        
        targets_dict = self.assign_targets(
            gt_boxes=data_dict['gt_boxes']
        )
        self.forward_ret_dict.update(targets_dict)

        ### Visualization
        # points = data_dict['points']    # (N, 5) [B, x, y, z, r]
        # batch_size = data_dict['batch_size']

        # cylind_range = self.cylind_range

        # resolution_w = 0.05 
        # resolution_h = 0.0021 

        # heatmaps = self.forward_ret_dict['heatmaps'][0]
        # W, H = heatmaps.shape[3] * 4, heatmaps.shape[2] * 4
        # import cv2

        # bev_frames = np.zeros((heatmaps.shape[0], H, W, 3)).astype(np.float64)
        
        # bevs_cylind = bev_frames

        # for i in range(batch_size):
        #     pts = points[points[:, 0] == i][:, 1:5].cpu().numpy().astype(np.float64)

        #     rho = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)  
        #     theta = np.arctan2(pts[:, 1], pts[:, 0])
        #     phz = np.stack((rho, theta, pts[:, 2]), axis=1)

        #     # generate bev map
        #     bev_xyz = convert_lidar_coords_to_image_coords(pts, W, H, 
        #         np.array([0, -40]), 0.05, 0.05, color=[255, 255, 255])
        #     bev = convert_lidar_coords_to_image_coords(phz, W, H, 
        #         cylind_range[:2], resolution_w, resolution_h, color=[255, 255, 255])
            
        #     bevs_cylind[i] = bev
        #     cv2.imshow('bev', bev)

        # # New img without points
        # # bev_frames = np.zeros((heatmaps.shape[0], W, H, 3)).astype(np.float32)
        # bev_frames = bevs_cylind
        # for i in range(heatmaps.shape[0]):
        #     num_pos = heatmaps[i]
        #     # print('num_pos %d: ' % i, num_pos)
        #     heatmap=np.array(heatmaps[i].permute(1,2,0).cpu())

        #     pos = (heatmap > 0)

        #     hm = np.zeros(heatmap.shape)
        #     hm[pos] = heatmap[pos]

        #     hm_tensor = torch.from_numpy(hm.transpose((2,1,0))).type(torch.FloatTensor).unsqueeze(1)
        #     up = nn.Upsample(scale_factor=(4,4))
        #     hm_up = up(hm_tensor)
        #     hm_up_array = hm_up.squeeze(1).numpy().transpose((2, 1, 0))
        #     indices = np.where(hm_up_array > 0)
        #     bev_frames[i][indices[:2]] = hm_up_array[indices[:2]] * 255

            # cv2.imshow('heatmap', (bev_frames[i] * 255).astype(np.uint8))
            # cv2.waitKey(0)
        
        # use box coords in xyz axis for visualization (B, max_obj, 7) [x, y, z, l, w, h, heading] # check if the dims are in order
        # gt_boxes = self.forward_ret_dict['anno_boxes_origin'][0].cpu().numpy()
        # bev_corners = np.zeros((gt_boxes.shape[0], gt_boxes.shape[1], 4, 2))
        # for i, box in enumerate(gt_boxes):
        #     corners_bev = box_utils.boxes3d_lidar_to_corners_bev(box)   # (M, 4, 3)
        #     bev_corners[i] = corners_bev[:, :, :2]

        # bev_corners_eight = np.zeros((gt_boxes.shape[0], gt_boxes.shape[1], 8, 2))
        # for i, box in enumerate(gt_boxes):
        #     # print(box.shape)
        #     corners_bev = box_utils.boxes3d_lidar_to_corners_bev(box, mode=1)   # (M, 8, 3)
        #     bev_corners_eight[i] = corners_bev[:, :, :2]
        # bev_corners = bev_corners_eight
        
        # bev_corners_in_cart = np.zeros_like(bev_corners).astype(np.int)
        # bev_corners_in_cart[:, :, :, 0] = ((bev_corners[:, :, :, 0] - 0) / 0.05)
        # bev_corners_in_cart[:, :, :, 1] = ((bev_corners[:, :, :, 1] + 40) / 0.05)

        # bev_corners_in_cylind = np.zeros_like(bev_corners).astype(np.int)   # (B, M, 4, 2)
        # bev_corners_in_cylind[:, :, :, 0] = (np.sqrt(bev_corners[:, :, :, 0] ** 2 + bev_corners[:, :, :, 1] ** 2) - cylind_range[0]) / resolution_w
        # bev_corners_in_cylind[:, :, :, 1] = (np.arctan2(bev_corners[:, :, :, 1], bev_corners[:, :, :, 0])  - cylind_range[1]) / resolution_h

        # rho_min_index = np.argmin(bev_corners_in_cylind[:, :, :, 0], axis=2)    # (B, M)
        # nearest_corners = np.take_along_axis(bev_corners_in_cylind[:,:,:,:], rho_min_index[..., np.newaxis, np.newaxis], axis=2).squeeze(2)     # (B, M, 2)

        # color = (255, 255, 0)

        # for batch_bev, batch_corners, batch_xyz_corners in zip(bev_frames, bev_corners_in_cylind, bev_corners_in_cart):
        #     bev_np = batch_bev
        #     for corners in batch_corners:
        #         corners_num = batch_corners.shape[1]   # 4
        #         for i in range(0, corners_num - 1):
        #             cv2.line(bev_np, corners[i], corners[i + 1], color, 2)
        #             # cv2.imshow('bev_with_box', bev_np)
        #             # cv2.waitKey()
        #         cv2.line(bev_np, corners[corners_num - 1], corners[0], color, 1)

        #         # p0, p1, p2, p3 = corner[0], corner[1], corner[2], corner[3]     # (4, 2)

        #         # cv2.line(bev_np, p0, p1, color, 2)
        #         # cv2.line(bev_np, p1, p2, color, 2)
        #         # cv2.line(bev_np, p2, p3, color, 2)
        #         # cv2.line(bev_np, p3, p0, color, 2)

        #     for corners in batch_xyz_corners:
        #         corners_num = batch_corners.shape[1]   # 4
        #         for i in range(0, corners_num - 1):
        #             cv2.line(bev_xyz, corners[i], corners[i + 1], color, 1)
        #             # cv2.imshow('bev_xyz', bev_xyz)
        #             # cv2.waitKey()

        #         cv2.line(bev_xyz, corners[corners_num - 1], corners[0], color, 1)


        #     for i in range(batch_corners.shape[0]):
        #         cv2.putText(bev_np, str(gt_boxes[0][i][-2].round(3)), nearest_corners[0][i] + 10, 1,1, (0,255,255), 2)
        #         cv2.putText(bev_xyz, str(gt_boxes[0][i][-2].round(3)), nearest_corners[0][i] + 10, 1,1, (0,255,255), 2)
        #         cv2.putText(bev_np, str(gt_boxes[0][i][-1].round(3)), nearest_corners[0][i], 1,1, (0,0,255), 2)
        #     cv2.namedWindow('bev_with_box', cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow('bev_with_box', 1100, 1000)
        #     cv2.imshow('bev_with_box', bev_np.astype(np.uint8))


        #     cv2.namedWindow('bev_xyz', cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow('bev_xyz', 1100, 1000)
        #     cv2.imshow('bev_xyz', bev_xyz.astype(np.uint8))
        #     cv2.waitKey()

        ### Visualization end.

        # flat_box_preds = box_preds.view(box_preds.shape[0], -1, box_preds.shape[-1]).clone()
        # batch_size = data_dict['batch_size']
        # for i in range(batch_size):
        #    targets_dict['anno_boxes'][0][i][..., -1] = targets_dict['anno_boxes'][0][i][..., -1] + 0.001
        #    flat_box_preds[i, targets_dict['inds'][0][i]] = targets_dict['anno_boxes'][0][i]
        
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=None
                #cls_preds=cls_preds, box_preds=flat_box_preds.view(box_preds.shape), dir_cls_preds=None
            )
            #data_dict['batch_cls_preds'] = targets_dict['heatmaps'][0].view(batch_size, 3, -1).permute(0, 2, 1)
            # print(data_dict['batch_cls_preds'][0][89219])
            # print(batch_box_preds[0][89219])
            # import pdb
            # pdb.set_trace()
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False
            # data_dict['cls_preds_normalized'] = True

        return data_dict

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor): Mask of the feature map with the shape
                of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)

        invalid_inds = torch.nonzero(ind == -1)
        invalid_inds_tuple = tuple(
            (invalid_inds[:, 0], invalid_inds[:, 1], invalid_inds[:, 2]))

        ind[invalid_inds_tuple] = 0
        feat = feat.gather(1, ind)
        feat[invalid_inds_tuple] = 0

        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def assign_targets(self, gt_boxes):
        """Generate targets.

        Args:
            gt_boxes: (B, M, 8) box + cls 

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including \
                    the following results in order.

                    - list[torch.Tensor]: Heatmap scores.
                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the \
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which \
                        boxes are valid.
        """
        gt_bboxes_3d, gt_labels_3d = gt_boxes[..., :-1], gt_boxes[..., -1]
        device = gt_bboxes_3d.device

        heatmaps, anno_boxes, inds, masks, anno_boxes_origin = multi_apply(
            self.get_targets_single, gt_bboxes_3d.to(device='cpu'), gt_labels_3d.to(device='cpu'))

        if len(heatmaps) > 1: 
            heatmaps = np.array(heatmaps).transpose(1, 0).tolist()
            heatmaps = [torch.stack(hms_).to(device) for hms_ in heatmaps]
            # transpose anno_boxes
            anno_boxes = np.array(anno_boxes).transpose(1, 0).tolist()
            anno_boxes = [torch.stack(anno_boxes_).to(device)
                        for anno_boxes_ in anno_boxes]

            anno_boxes_origin = np.array(anno_boxes_origin).transpose(1, 0).tolist()
            anno_boxes_origin = [torch.stack(anno_boxes_).to(device)
                        for anno_boxes_ in anno_boxes_origin]
            # transpose inds
            inds = np.array(inds).transpose(1, 0).tolist()
            inds = [torch.stack(inds_).to(device) for inds_ in inds]
            # transpose masks
            masks = np.array(masks).transpose(1, 0).tolist()
            masks = [torch.stack(masks_).to(device) for masks_ in masks]
        else:
            heatmaps = [[heatmaps[0][0]]]
            heatmaps = [torch.stack(hms_).to(device) for hms_ in heatmaps]
            # transpose anno_boxes
            anno_boxes = [[anno_boxes[0][0]]]
            anno_boxes = [torch.stack(anno_boxes_).to(device)
                        for anno_boxes_ in anno_boxes]

            anno_boxes_origin = [[anno_boxes_origin[0][0]]]
            anno_boxes_origin = [torch.stack(anno_boxes_).to(device)
                        for anno_boxes_ in anno_boxes_origin]

            # transpose inds
            inds = [[inds[0][0]]]
            inds = [torch.stack(inds_).to(device) for inds_ in inds]
            # transpose masks
            masks = [[masks[0][0]]]
            masks = [torch.stack(masks_).to(device) for masks_ in masks]

        # import pdb
        # pdb.set_trace() 
        # num_pos = heatmaps[0].eq(1).float().sum().item()
        # print('num_pos: ', num_pos)

        # transpose heatmaps, because the dimension of tensors in each task is
        # different, we have to use numpy instead of torch to do the transpose.
        # import cv2
        # batch_size = heatmaps[0].shape[0]
        # for i in range(batch_size):
        #     heatmap = heatmaps[0][i]
        #     num_pos = heatmap.eq(1).float().sum().item()
        #     print('num_pos_%d: '%i, num_pos)

        #     heatmap=np.array(heatmap.cpu()) * 255
        #     heatmap=heatmap.astype(np.uint8).transpose(1, 2, 0)

        #     # heatmap=cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        #     cv2.imwrite('heatmap_%d_interp.png' % i,heatmap)
        

        # cv2.waitKey(0)

        # import pdb
        # pdb.set_trace()

        all_targets_dict = {
            'heatmaps': heatmaps,
            'anno_boxes': anno_boxes,
            'inds': inds,
            'masks': masks,
            'anno_boxes_origin': anno_boxes_origin,
        }

        return all_targets_dict

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        device = gt_labels_3d.device
        """gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
        """
        max_objs = self.target_cfg.MAX_OBJS
        
        grid_size = torch.tensor(self.cy_grid_size) if self.cy_grid_size is not None else torch.tensor(self.grid_size)
        
        pc_range = torch.tensor(self.point_cloud_range)
        voxel_size = torch.tensor(self.target_cfg.VOXEL_SIZE)

        cylind_range = torch.tensor(self.cylind_range) if self.cylind_range is not None else None
        cylind_size = torch.tensor(self.target_cfg.CYLIND_SIZE) if hasattr(self.target_cfg, 'CYLIND_SIZE') else None
        
        feature_map_size = grid_size[:2] // self.target_cfg.OUT_SIZE_FACTOR
        feature_map_size = feature_map_size.to(device, dtype=torch.int32)
        
        """
        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            print(gt_labels_3d)
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        """

        # print('gt_bboxes: ', gt_bboxes_3d[0])

        task_boxes = [gt_bboxes_3d]
        task_classes = [gt_labels_3d]
        
        draw_gaussian = draw_heatmap_gaussian_cylind if self.cylind else draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks, anno_boxes_origin = [], [], [], [], []

        for idx in range(1):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))

            anno_box = gt_bboxes_3d.new_zeros((max_objs, self.num_reg_channels),
                                              dtype=torch.float32)
            
            anno_box_origin = gt_bboxes_3d.new_zeros((max_objs, self.num_reg_channels),
                                    dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = (task_classes[idx][k] - 1).int()

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.target_cfg.OUT_SIZE_FACTOR
                length = length / voxel_size[1] / \
                    self.target_cfg.OUT_SIZE_FACTOR

                if width > 0 and length > 0:
                    radius = gaussian_radius((length, width), min_overlap=self.target_cfg.GAUSSIAN_OVERLAP)
                    radius = max(self.target_cfg.MIN_RADIUS, int(radius))
                    # if self.target_cfg.MIN_RADIUS < int(radius):
                    #     import pdb
                    #     pdb.set_trace()
                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    if self.cylind:
                        rho = np.sqrt(x ** 2 + y ** 2)
                        phi = np.arctan2(y, x)

                        coor_x = (
                            rho - cylind_range[0]
                        ) / cylind_size[0] / self.target_cfg.OUT_SIZE_FACTOR
                        coor_y = (
                            phi - cylind_range[1]
                        ) / cylind_size[1] / self.target_cfg.OUT_SIZE_FACTOR

                    else:
                        coor_x = (
                            x - pc_range[0]
                        ) / voxel_size[0] / self.target_cfg.OUT_SIZE_FACTOR
                        coor_y = (
                            y - pc_range[1]
                        ) / voxel_size[1] / self.target_cfg.OUT_SIZE_FACTOR

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.floor().to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue
                    
                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    if self.cylind: 
                        rho_center = x * cylind_size[0] * self.target_cfg.OUT_SIZE_FACTOR + cylind_range[0]
                        center_l = cylind_size[1] * rho_center
                        y_factor = cylind_size[0]  / center_l

                        if y_factor < 2:
                            y_factor = 2
                        draw_gaussian(heatmap[cls_id], center_int, radius, y_factor=y_factor, 
                            resolution=self.target_cfg.CYLIND_SIZE[:2], min_range_rho=self.cylind_range[0])
                    else:
                        draw_gaussian(heatmap[cls_id], center_int, radius)

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = y * feature_map_size[0] + x

                    mask[new_idx] = 1
                    heading = task_boxes[idx][k][6]

                    box_dim = task_boxes[idx][k][3:6]
                    box_dim = box_dim.log()
                    height = task_boxes[idx][k][5:6]
                    height_dim = height.log()
                
                    if self.cylind:
                        center_arctan = (y * self.target_cfg.OUT_SIZE_FACTOR * cylind_size[1] + cylind_range[1])
                        arc = phi - center_arctan
                        r = x * self.target_cfg.OUT_SIZE_FACTOR * cylind_size[0] + cylind_range[0]
                        # same as arc
                        # offset_arc = (center[1] - y) * (cylind_size[1] * self.target_cfg.OUT_SIZE_FACTOR)

                        rot_rel = heading.unsqueeze(0) - center_arctan

                        # anno_box[new_idx] = torch.cat([
                        #     center[0: 1] -
                        #     torch.tensor([x], device=device,
                        #                 dtype=torch.float32),
                        #     arc * r.unsqueeze(0),
                        #     #center[1: 2] -
                        #     #torch.tensor([y], device=device,
                        #     #            dtype=torch.float32),
                        #     z.unsqueeze(0), box_dim,
                        #     torch.sin(rot_rel),
                        #     torch.cos(rot_rel),
                        # ])

                        gt_box = task_boxes[idx][k:k+1] # (1,7)

                        corners_bev = box_utils.boxes_to_corners_3d(gt_box)   # (1, 8, 3)
                        bev_corners = corners_bev[0, :4, :2]

                        cylind_bev_corners = torch.zeros_like(bev_corners)  # (4, 2)
                        cylind_bev_corners[:, 0] = (torch.sqrt(bev_corners[:, 0] ** 2 + bev_corners[:, 1] ** 2) - 
                                                    cylind_range[0]) / cylind_size[0] / self.target_cfg.OUT_SIZE_FACTOR
                        cylind_bev_corners[:, 1] = (torch.atan2(bev_corners[:, 1], bev_corners[:, 0]) 
                                                    - cylind_range[1]) / cylind_size[1] / self.target_cfg.OUT_SIZE_FACTOR

                        nearest_index = torch.argmin(cylind_bev_corners[:, 0])    # (1)
                        nearest_corner = cylind_bev_corners[nearest_index].squeeze(0)     # (2)
                        corner_offset = nearest_corner - center
                        
                        
                        anno_box[new_idx] = torch.cat([
                            center[0: 1] -
                            torch.tensor([x], device=device,
                                        dtype=torch.float32),
                            arc * r.unsqueeze(0),
                            z.unsqueeze(0), 
                            corner_offset,
                            height_dim,
                            torch.sin(rot_rel),
                            torch.cos(rot_rel),
                        ])

                        # center[1] = arc * r / rho + center[1]
                        # corner = corner_offset + center

                        # import pdb
                        # pdb.set_trace()
                        # corner_rho = corner[0] * self.target_cfg.OUT_SIZE_FACTOR * cylind_size[0] + cylind_range[0]
                        # corner_phi = corner[1] * self.target_cfg.OUT_SIZE_FACTOR * cylind_size[1] + cylind_range[1]
                        # print('corner_rho: ', corner_rho, corner_phi)
                        # corner_xy = torch.stack((corner_rho * torch.cos(corner_phi), corner_rho * torch.sin(corner_phi)))
                        # print('corner_xy: ', corner_xy)
                        # rotate_matrix=torch.tensor([[torch.cos(-heading), -torch.sin(-heading), 0],
                        #         [np.sin(-heading), np.cos(-heading), 0],
                        #         [0, 0, 1]])  # (3, 3, 3) 
                        # trans_corner = torch.mm(rotate_matrix, 
                        #                 torch.tensor([corner_xy[0] - task_boxes[idx][k][0], corner_xy[1] - task_boxes[idx][k][1], 1]).unsqueeze(1))
                        # print('corner_offset: ', corner_xy[0] - task_boxes[idx][k][0], corner_xy[1] - task_boxes[idx][k][1])
                        # print('trans_corner: ', trans_corner)
                        # length = torch.abs(trans_corner[0] * 2)
                        # width = torch.abs(trans_corner[1] * 2)
                        # print('length: ', length, ' width: ', width)
                        # print('gt:', task_boxes[idx][k][3:5])
                        
                        # print('rot_rel: ', rot_rel, ' heading: ', heading, 'center_arctan: ', center_arctan)
                        anno_box_origin[new_idx] = torch.cat([
                            task_boxes[idx][k][0:2],
                            z.unsqueeze(0), task_boxes[idx][k][3:6],
                            heading.unsqueeze(0), rot_rel, 
                        ])

                    else:
                        anno_box[new_idx] = torch.cat([
                            center -
                            torch.tensor([x, y], device=device,
                                        dtype=torch.float32),
                            z.unsqueeze(0), box_dim,
                            torch.sin(heading).unsqueeze(0),
                            torch.cos(heading).unsqueeze(0),
                        ])

            # import cv2
            # heatmap2=np.array(heatmap[0].cpu()) * 255
            # heatmap2=heatmap2.astype(np.uint8)
            # # heatmap2=cv2.applyColorMap(heatmap2, cv2.COLORMAP_HOT)
            # cv2.imshow('heatmap_with_num',heatmap2)
            # cv2.waitKey(0)

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            if self.cylind:
                anno_boxes_origin.append(anno_box_origin)
            masks.append(mask)
            inds.append(ind)

        return heatmaps, anno_boxes, inds, masks, anno_boxes_origin

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        batch, H, W, code_size = box_preds.size()
        box_preds = box_preds.reshape(batch, H*W, code_size)

        batch_reg = box_preds[..., 0:2]
        batch_hei = box_preds[..., 2:3]

        batch_dim = torch.exp(box_preds[..., 3:6])

        yc, xc = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        yc = yc.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device).view(batch, -1, 1).float()
        xc = xc.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device).view(batch, -1, 1).float()

        voxel_cx = xc.clone()
        voxel_cy = yc.clone()

        if self.cylind:
            cylind_range = torch.tensor(self.cylind_range)
            cylind_size = torch.tensor(self.target_cfg.CYLIND_SIZE) 
            
            rho = (voxel_cx + batch_reg[:, :, 0:1]) * self.target_cfg.OUT_SIZE_FACTOR * \
                cylind_size[0] + cylind_range[0]
            
            voxel_phi = voxel_cy * self.target_cfg.OUT_SIZE_FACTOR * \
                cylind_size[1] + cylind_range[1]

            angle_offset = batch_reg[:, :, 1:2] / rho   # in rad

            phi = voxel_phi + angle_offset

            box_cx = (rho * torch.cos(phi))  # (B, M, 1)
            box_cy = (rho * torch.sin(phi))
            
            # print('cx: ', box_cx[0, 89219], box_cy[0, 89219])

            batch_rots = box_preds[..., 6:7]
            batch_rotc = box_preds[..., 7:8]
            heading = torch.atan2(batch_rots, batch_rotc) + voxel_phi
            
            # print('===================================================')
            # print('gen heading: ', heading[0, 89219])

            height_dim = torch.exp(box_preds[..., 5:6])
            
            corner_offset = box_preds[..., 3:5]
            corner = corner_offset

            corner_rho = (corner[:, :, 0:1] + voxel_cx + batch_reg[:, :, 0:1]) * self.target_cfg.OUT_SIZE_FACTOR * cylind_size[0] + cylind_range[0]
            corner_phi = (corner[:, :, 1:2] + voxel_cy) * self.target_cfg.OUT_SIZE_FACTOR * cylind_size[1] + cylind_range[1] + angle_offset

            corner_xy = torch.cat((corner_rho * torch.cos(corner_phi), corner_rho * torch.sin(corner_phi)), axis=2).view(-1, 2)
            # print('corner rho: ', corner_rho[0, 89219], corner_phi[0,89219])
            ry = heading.view(-1, 1)

            corner_vector = torch.stack([corner_xy[:, 0:1] - box_cx.view(-1, 1), corner_xy[:, 1:2] - box_cy.view(-1, 1) , torch.ones_like(box_cx.view(-1, 1) )], axis=1)

            rotate_matrix = torch.zeros(ry.shape[0], 3, 3).to(ry.device)
            rotate_matrix[:, 0, 0:1] = torch.cos(-ry)
            rotate_matrix[:, 0, 1:2] = -torch.sin(-ry)
            rotate_matrix[:, 1, 0:1] = torch.sin(-ry)
            rotate_matrix[:, 1, 1:2] = torch.cos(-ry)
            rotate_matrix[:, 2, 2:3] = torch.ones_like(ry)  # (B, M, 3, 3, 3) 

            trans_corner = torch.bmm(rotate_matrix, corner_vector).reshape(batch, -1, 3)
            length = torch.abs(trans_corner[:, :, 0:1] * 2)
            width = torch.abs(trans_corner[:, :, 1:2] * 2)
   
            # import pdb
            # pdb.set_trace()

            batch_dim = torch.cat([length, width, height_dim], axis=2)
        else:   
            box_cy = voxel_cy + batch_reg[:, :, 1:2]
            box_cx = (voxel_cx + batch_reg[:, :, 0:1]) * self.target_cfg.OUT_SIZE_FACTOR * \
                self.target_cfg.VOXEL_SIZE[0] + self.point_cloud_range[0]
            box_cy = box_cy * self.target_cfg.OUT_SIZE_FACTOR * \
                self.target_cfg.VOXEL_SIZE[1] + self.point_cloud_range[1]
        
            batch_rots = box_preds[..., 6:7]
            batch_rotc = box_preds[..., 7:8]
            heading = torch.atan2(batch_rots, batch_rotc) 

        batch_box_preds = torch.cat([box_cx, box_cy, batch_hei, batch_dim, heading], dim=2)
        batch_cls_preds = cls_preds.view(batch, H*W, -1)
        # 89219
        #                   xs,       ys,      batch_hei, batch_dim,               rot
        # box_preds:        [0.5785,  0.2246, -0.7503,  1.3974,  0.4941,  0.4284,  2.8610]
        # batch_box_preds:  [58.8903, 16.3419, -0.7503,  4.0448,  1.6391,  1.5348,  3.1298]
        # anno_boxes:       [0.3442,  0.6907, -0.8411,  1.3056,  0.6259,  0.5128, -3.4096]
        # targets_dict['anno_boxes_origin'][0][0][0]: [58.7808, 16.5596, -0.8411,  3.6900,  1.8700,  1.6700, -3.1408]
        # print(data_dict['batch_box_preds'][0][89219]) [58.8759, 16.2183, -0.8411,  3.6900,  1.8700,  1.6700,  3.1366]
        #print('box: ', batch_box_preds[0][89219], ' phi: ', phi[0][89219])
        #import pdb
        #pdb.set_trace()
        return batch_cls_preds, batch_box_preds

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss
        # print('cls_loss: ', cls_loss.item(), ' box_loss: ', box_loss.item())

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def get_cls_layer_loss(self):
        # NHWC -> NCHW
        pred_heatmaps = clip_sigmoid(
            self.forward_ret_dict['cls_preds']).permute(0, 3, 1, 2)
        device = pred_heatmaps.device
        gt_heatmaps = self.forward_ret_dict['heatmaps'][0].to(device)
        num_pos = gt_heatmaps.ge(1).float().sum().item()

        # print('num_pos: ', num_pos)
 
        # import cv2
        # for i in range(gt_heatmaps.shape[0]):
        #     num_pos = gt_heatmaps[i].eq(1).float().sum().item()
        #     print('num_pos %d: ' % i, num_pos)
        #     heatmap=np.array(gt_heatmaps[i].permute(1,2,0).cpu()) * 255
        #     heatmap=heatmap.astype(np.uint8)
        #     #heatmap=cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        #     cv2.imshow('heatmap',heatmap)
        #     cv2.waitKey(0)

        cls_loss = self.loss_cls(
            pred_heatmaps,
            gt_heatmaps,
            avg_factor=max(num_pos, 1))

        cls_loss = cls_loss * \
            self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }

        # print('cls_loss:', cls_loss)
        return cls_loss, tb_dict

    def get_box_reg_layer_loss(self):
        # Regression loss for dimension, offset, height, rotation
        target_box, inds, masks = self.forward_ret_dict['anno_boxes'][
            0], self.forward_ret_dict['inds'][0], self.forward_ret_dict['masks'][0]

        ind = inds
        num = masks.float().sum()
        pred = self.forward_ret_dict['box_preds']  # B x (HxW) x 8
        pred = pred.view(pred.size(0), -1, pred.size(3))
        pred = self._gather_feat(pred, ind)


        # if self.forward_ret_dict['heatmaps'][0].ge(1).float().sum().item() != num.item():
        #     print('ht: ', self.forward_ret_dict['heatmaps'][0].ge(1).float().sum().item(), ' num: ', num.item())
        #     pdb.set_trace()
        mask = masks.unsqueeze(2).expand_as(target_box).float()
        isnotnan = (~torch.isnan(target_box)).float()
        mask *= isnotnan

        code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
        
        bbox_weights = mask * mask.new_tensor(code_weights[:pred.shape[2]])

        # rcnn_batch_size = 
        # reg_targets = target_box.view(rcnn_batch_size, -1)
        # rcnn_loss_reg = F.l1_loss(
        #     rcnn_reg.view(rcnn_batch_size, -1),
        #     reg_targets,
        #     reduction='none'
        # )  # [B, M, 7]

        # rcnn_loss_reg = rcnn_loss_reg * rcnn_loss_reg.new_tensor(\
        #     loss_cfgs.LOSS_WEIGHTS['code_weights'])

        # rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
        
        loc_loss = l1_loss(
            pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
        
        xy_loss = l1_loss(
            pred[:, :, 0:2], target_box[:, :, 0:2], bbox_weights[:, :, 0:2], avg_factor=(num + 1e-4)
        )
        z_loss = l1_loss(
            pred[:, :, 2:3], target_box[:, :, 2:3], bbox_weights[:, :, 2:3], avg_factor=(num + 1e-4)
        )
        dim_loss = l1_loss(
            pred[:, :, 3:6], target_box[:, :, 3:6], bbox_weights[:, :, 3:6], avg_factor=(num + 1e-4)
        )
        yaw_loss = l1_loss(
            pred[:, :, 6:], target_box[:, :, 6:], bbox_weights[:, :, 6:], avg_factor=(num + 1e-4))
        
        #print('x_loss: ', x_loss.item())
        #print('y_loss: ', y_loss.item())
        #print('dim_loss: ', dim_loss.item())
        #print('loc_loss: ', loc_loss.item())
        #print('yaw_loss: ', yaw_loss.item())

        loc_loss = loc_loss * \
            self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item(),
            'rpn_loss_yaw': yaw_loss.item(),
            'rpn_loss_dim': dim_loss.item(),
            'rpn_loss_xy': xy_loss.item(),
            'rpn_loss_z': z_loss.item()
        }

        return box_loss, tb_dict


"""
The following is some util files, we will move it to separate files later
"""


def clip_sigmoid(x, eps=1e-4):
    """Sigmoid function for input feature.

    Args:
        x (torch.Tensor): Input feature map with the shape of [B, N, H, W].
        eps (float): Lower bound of the range to be clamped to. Defaults
            to 1e-4.

    Returns:
        torch.Tensor: Feature map after sigmoid.
    """
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


def gaussian_2d(shape, sigma=1, sigma2=None):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    if sigma2 is not None:
        sigma1 = sigma
        sigma2 = sigma2
        h = np.exp(-(x * x  / (2 * sigma1 * sigma1) + y * y / (2 * sigma2 * sigma2)))
    else:
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_2d_cylind(shape, center, resolution, scale, sigma=1, min_range_rho=0):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map. [y, x]
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map. [rho, phi]
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    
    rho_res, theta_res = resolution[0], resolution[1]
    rho = (x + center[0]) * rho_res * scale + min_range_rho
    theta_offset = y * theta_res * scale
    # h = np.exp(-(rho_center * rho_center + (rho_center + rho_offset) ** 2 
    #     - 2 * rho_center * (rho_center + rho_offset) * np.cos(theta_offset) ) / (2 * sigma * sigma))
    
    rho_center = rho[:, (shape[1] - 1) // 2]

    distance_square = rho ** 2 + rho_center ** 2 - 2 * rho * rho_center * np.cos(theta_offset)

    h = np.exp(-(distance_square / ((rho_res * scale) ** 2) / (2 * sigma * sigma)))

    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1, y_factor=1, resolution=[0.05, 0.0021], scale=4):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked. shape (phi, rho)
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gausian.
        K (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    try:
        y_radius = int(((radius * y_factor).floor().item()))
    except:
        y_radius = radius
    y_diameter = 2 * y_radius + 1
    # gaussian_center = gaussian_2d((diameter, diameter), sigma=diameter / 6).transpose(1, 0)

    # index = np.linspace(0, diameter - 1, diameter, endpoint=True)
    # y_index = radius  / (y_radius + 1) * np.linspace(0, y_radius + 1, y_radius + 1, endpoint=True)
    # gaussian = np.zeros((y_diameter, diameter))

    # for i in range(gaussian_center.shape[0]):
    #     gaussian_half = np.interp(y_index, index, gaussian_center[i])
    #     gaussian[:, i] = np.concatenate((gaussian_half, gaussian_half[:-1][::-1]), axis=0)

    gaussian = gaussian_2d((y_diameter, diameter), sigma=diameter / 6, sigma2 = y_diameter / 6)

    x, y = int(center[0]), int(center[1])
    
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, y_radius), min(height - y, y_radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[y_radius - top:y_radius + bottom,
                 radius - left:radius + right]).to(heatmap.device,
                                                   torch.float32)
    
    # gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6).transpose(1, 0)

    # x, y = int(center[0]), int(center[1])
    
    # height, width = heatmap.shape[0:2]

    # left, right = min(x, radius), min(width - x, radius + 1)
    # top, bottom = min(y, radius), min(height - y, radius + 1)

    # masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    # masked_gaussian = torch.from_numpy(
    #     gaussian[radius - top:radius + bottom,
    #              radius - left:radius + right]).to(heatmap.device,
    #                                                torch.float32) 

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    # if radius != y_radius:
    #     print('radius: ', radius, ' y_radius: ', y_radius)
    #     print('y_factor: ', y_factor, ' y: ', y)
    #     import pdb
    #     pdb.set_trace()
    #     import cv2
    #     heatmap2=np.array(heatmap.cpu()) * 255
    #     heatmap2=heatmap2.astype(np.uint8)
    #     heatmap2=cv2.applyColorMap(heatmap2, cv2.COLORMAP_HOT)
    #     cv2.imshow('heatmap',heatmap2)
    #     cv2.waitKey(0)
    return heatmap



def draw_heatmap_gaussian_cylind(heatmap, center, radius, 
    k=1, y_factor=1, resolution=[0.05, 0.0021], scale=4, min_range_rho=0):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked. shape (phi, rho)
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gausian.
        K (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    try:
        y_radius = int(((radius * y_factor).floor().item()))
    except:
        y_radius = radius
    y_diameter = 2 * y_radius + 1
    # gaussian_center = gaussian_2d((diameter, diameter), sigma=diameter / 6).transpose(1, 0)

    # index = np.linspace(0, diameter - 1, diameter, endpoint=True)
    # y_index = radius  / (y_radius + 1) * np.linspace(0, y_radius + 1, y_radius + 1, endpoint=True)
    # gaussian = np.zeros((y_diameter, diameter))

    # for i in range(gaussian_center.shape[0]):
    #     gaussian_half = np.interp(y_index, index, gaussian_center[i])
    #     gaussian[:, i] = np.concatenate((gaussian_half, gaussian_half[:-1][::-1]), axis=0)

    # 2-dim gaussian distribution
    gaussian = gaussian_2d((y_diameter, diameter), sigma=diameter / 6, sigma2 = y_diameter / 6)

    x, y = int(center[0]), int(center[1])
    
    height, width = heatmap.shape[0:2]
    
    # gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6).transpose(1, 0)

    # x, y = int(center[0]), int(center[1])
    
    # height, width = heatmap.shape[0:2]

    # left, right = min(x, radius), min(width - x, radius + 1)
    # top, bottom = min(y, radius), min(height - y, radius + 1)

    # masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    # masked_gaussian = torch.from_numpy(
    #     gaussian[radius - top:radius + bottom,
    #              radius - left:radius + right]).to(heatmap.device,
    #                                                torch.float32) 

    cy_radius_x = radius
    cy_radius_y = int(math.acos(1 - 0.5 * (radius / x) ** 2) / resolution[1] / scale)

    y_diameter = 2 * cy_radius_y + 1
    x_diameter = 2 * cy_radius_x + 1

    gaussian = gaussian_2d_cylind((y_diameter, x_diameter), sigma=(2 * radius + 1) / 6, 
            center=[x,y], resolution=resolution, scale=scale, min_range_rho=min_range_rho)

    left, right = min(x, cy_radius_x), min(width - x, cy_radius_x + 1)
    top, bottom = min(y, cy_radius_y), min(height - y, cy_radius_y + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[cy_radius_y - top:cy_radius_y + bottom,
                 cy_radius_x - left:cy_radius_x + right]).to(heatmap.device,
                                                   torch.float32)

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    # Visualization of heatmap
    # print('radius: {}, rho_radius: {}, theta_radius: {}'.format(radius, cy_radius_x, cy_radius_y))
    # print('gaussian shape: ', gaussian.shape)

    # import cv2
    # heatmap2=np.array(heatmap.cpu()) * 255
    # heatmap2=heatmap2.astype(np.uint8)
    # # heatmap2=cv2.applyColorMap(heatmap2, cv2.COLORMAP_HOT)
    # cv2.imshow('heatmap',heatmap2)
    # cv2.waitKey(0)

    return heatmap


def gaussian_radius(det_size, min_overlap=0.5):
    """Get radius of gaussian.

    Args:
        det_size (tuple[torch.Tensor]): Size of the detection result.
        min_overlap (float): Gaussian_overlap. Defaults to 0.5.

    Returns:
        torch.Tensor: Computed radius.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


"""
Gaussian Loss 
"""


class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negtive samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_reg = self.loss_weight * gaussian_focal_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_reg


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


@weighted_loss
def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1).float()
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss


@weighted_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss
