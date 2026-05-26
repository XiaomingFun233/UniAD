#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import pickle
from mmdet.models import LOSSES


@LOSSES.register_module()
class PlanningLoss(nn.Module):
    def __init__(self, loss_type='L2'):
        super(PlanningLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, sdc_traj, gt_sdc_fut_traj, mask):
        err = sdc_traj[..., :2] - gt_sdc_fut_traj[..., :2]
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=-1)
        err = torch.pow(err, exponent=0.5)
        return torch.sum(err * mask)/(torch.sum(mask) + 1e-5)


@LOSSES.register_module()
class CollisionLoss(nn.Module):
    def __init__(self, delta=0.5, weight=1.0):
        super(CollisionLoss, self).__init__()
        self.w = 1.85 + delta
        self.h = 4.084 + delta
        self.weight = weight
    
    def forward(self, sdc_traj_all, sdc_planning_gt, sdc_planning_gt_mask, future_gt_bbox):
        # sdc_traj_all (1, 6, 2)
        # sdc_planning_gt (1,6,3)
        # sdc_planning_gt_mask (1, 6)
        # future_gt_bbox 6x[lidarboxinstance]
        n_futures = len(future_gt_bbox)
        inter_sum = sdc_traj_all.new_zeros(1, )
        for i in range(n_futures):
            if len(future_gt_bbox[i].tensor) > 0:
                future_gt_bbox_corners = future_gt_bbox[i].corners[:, [0,3,4,7], :2] # (N, 8, 3) -> (N, 4, 2) only bev 
                # sdc_yaw = -sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype) - 1.5708
                sdc_yaw = sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype)
                sdc_bev_box = self.to_corners([sdc_traj_all[0, i, 0], sdc_traj_all[0, i, 1], self.w, self.h, sdc_yaw])  
                for j in range(future_gt_bbox_corners.shape[0]):
                    inter_sum += self.inter_bbox(sdc_bev_box, future_gt_bbox_corners[j].to(sdc_traj_all.device))  
        return inter_sum * self.weight
        
    def inter_bbox(self, corners_a, corners_b):
        # Optimization principle:
        # This implementation computes the min/max bounds of each box along all coordinate
        # dimensions at once, instead of manually slicing and reducing x/y coordinates one
        # by one. By using vectorized tensor reductions and elementwise operations, it
        # reduces repeated Python-level indexing, duplicated reduction calls, and temporary
        # scalar tensor operations. The intersection box is derived from the smaller upper
        # bound and the larger lower bound, then negative side lengths are clamped to zero
        # before multiplying all dimensions. This makes the code faster, cleaner, and
        # naturally extensible to higher-dimensional bounding boxes.
        a_max = corners_a.max(dim=0).values
        a_min = corners_a.min(dim=0).values
        b_max = corners_b.max(dim=0).values
        b_min = corners_b.min(dim=0).values

        inter_max = torch.minimum(a_max, b_max)
        inter_min = torch.maximum(a_min, b_min)
        inter_wh = (inter_max - inter_min).clamp(min=0)
        intersect = inter_wh.prod()

        # The code above is the optimized version, and the code below is the original version.
        #xa1, ya1 = torch.max(corners_a[:, 0]), torch.max(corners_a[:, 1])
        #xa2, ya2 = torch.min(corners_a[:, 0]), torch.min(corners_a[:, 1])
        #xb1, yb1 = torch.max(corners_b[:, 0]), torch.max(corners_b[:, 1])
        #xb2, yb2 = torch.min(corners_b[:, 0]), torch.min(corners_b[:, 1])
        #
        #xi1, yi1 = torch.minimum(xa1, xb1), torch.minimum(ya1, yb1)
        #xi2, yi2 = torch.maximum(xa2, xb2), torch.maximum(ya2, yb2)
        #intersect = torch.clamp((xi1 - xi2), min=0) * torch.clamp((yi1 - yi2), min=0)
        return intersect

    def to_corners(self, bbox):
        x, y, w, l, theta = bbox
        corners = torch.tensor([
            [w/2, -l/2], [w/2, l/2], [-w/2, l/2], [-w/2,-l/2]  
        ]).to(x.device) # 4,2
        rot_mat = torch.tensor(
            [[torch.cos(theta), torch.sin(theta)],
             [-torch.sin(theta), torch.cos(theta)]]
        ).to(x.device)
        new_corners = rot_mat @ corners.T + torch.tensor(bbox[:2])[:, None].to(x.device)
        return new_corners.T
