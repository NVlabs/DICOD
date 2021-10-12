# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image 

def levels_to_images(mlvl_tensor):
        """Concat multi-level feature maps by image.
        """
        batch_size = len(mlvl_tensor[0])
        batch_list = [[] for _ in range(batch_size)]
        for t in mlvl_tensor:
            for img in range(batch_size):
                if t[img] is not None:
                    batch_list[img].append(t[img])
        return [torch.cat(item, 0) for item in batch_list]

def generate_gaps(cfg):
    gs = cfg.gap_size
    if cfg.gap == 'AdaptiveAvgPool':
        if -1 in cfg.layer:
            gaps = [nn.AdaptiveAvgPool2d((gs*(2**i), gs*(2**i))) for i in range(len(cfg.layer))]
        else:
            gaps = [nn.AdaptiveAvgPool2d((gs*(2**i), gs*(2**i))) for i in range(len(cfg.layer))]
    else:
        gaps = [None for i in range(len(cfg.layer))]
    return gaps

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

class LOC_STN(nn.Module):
    def __init__(self, out_size, loss_weight):
        super(LOC_STN, self).__init__()
        self.out_size = out_size
        self.loss_weight = loss_weight
        self.batch_size = 256
    
    def _get_objs(self, bboxs, input_img, device):
        height, width = input_img.size(2), input_img.size(3)
        left = torch.clamp(bboxs[ :, 0], min=0)
        bottom = torch.clamp(bboxs[:, 1], min=0)
        right = torch.clamp(bboxs[:, 2], max=width)
        top = torch.clamp(bboxs[:, 3], max=height)
        
        theta = torch.zeros((bboxs.size(0), 2, 3), dtype=torch.float, device=device) 
        
        theta[:, 0, 0] = (right-left)/width
        theta[:, 1, 1] = (top-bottom)/height
        
        theta[:, 0, 2] =  -1. +  (left + (right-left)/2)/(width/2)
        theta[:, 1, 2] =  -1. + ((bottom + (top-bottom)/2))/(height/2)
        
        grid_size = torch.Size((bboxs.size(0), input_img.size(1), self.out_size, self.out_size))
        grid = F.affine_grid(theta, grid_size, align_corners=False) # align_corners=True or False didn't affect too much
        objs = F.grid_sample(torch.cat(bboxs.size(0)*[input_img]), grid, align_corners=False) 

        return objs

    def _forward_pool(self, pos_bbox_pred, pos_bbox_targets, img, loc_teacher, pos_bbox_labels=None, loss=None, gap=None, layer=None):
        device = img.device
        pred_objs_total = []
        target_objs_total = []
        for i in range(img.size(0)):
            input_img= img[i].unsqueeze(0) # N C H W
            pos_bbox_pred_img = pos_bbox_pred[i]
            pos_bbox_targets_img = pos_bbox_targets[i]
        
            if pos_bbox_pred_img.size(0) < 1:
                continue
            else:
                pos_bbox_pred_img = pos_bbox_pred_img
                pos_bbox_targets_img = pos_bbox_targets_img
            
            pred_objs = self._get_objs(pos_bbox_pred_img, input_img, device)
            target_objs = self._get_objs(pos_bbox_targets_img, input_img, device)
            pred_objs_total.append(pred_objs)
            target_objs_total.append(target_objs)

        assert len(pred_objs_total) == len(target_objs_total)
        if len(pred_objs_total) > 0:
            pred_objs_all = torch.cat(pred_objs_total, dim=0).to(device) 
            target_objs_all = torch.cat(target_objs_total, dim=0).to(device)
            if pred_objs_all.size(0) > 0:
                gap_loss = self._get_feature_loss_batch(pred_objs_all, target_objs_all, loc_teacher, gap, loss, layer)
                distill_loc_loss = self.loss_weight * gap_loss
        else:
            distill_loc_loss = torch.tensor(0.).to(device)
        return distill_loc_loss
        

    def _get_feature_loss_batch(self, pred_objs_all, target_objs_all, loc_teacher, gap, loss, layers):
        """
        gap: a list
        """
        batch_size = self.batch_size
        cut = list(range(0, pred_objs_all.size(0), batch_size))
        cut += [pred_objs_all.size(0)]
        gap_loss = torch.tensor(0.)
        for i in range(len(cut) - 1):
            pred_objs = pred_objs_all[cut[i]: cut[i+1]]
            target_objs = target_objs_all[cut[i]: cut[i+1]]
            if loc_teacher is None:
                if gap[0] is None:
                    pred_objs_feats = pred_objs
                    target_objs_feats = target_objs
                else:
                    pred_objs_feats = gap[0](pred_objs)
                    target_objs_feats = gap[0](target_objs)
                assert len(pred_objs.size()) == 4
                if loss == 'l1':
                    gap_loss_ = torch.norm((pred_objs_feats - target_objs_feats), p=1)        
                    gap_loss = gap_loss + gap_loss_ /(pred_objs_feats.size(1)*pred_objs_feats.size(2) * pred_objs_feats.size(3))
            else:
                if layers[0] == -1:
                    if gap[0] is None:
                        pred_objs_feats = pred_objs
                        target_objs_feats = target_objs
                    else:
                        pred_objs_feats = gap[0](pred_objs)
                        target_objs_feats = gap[0](target_objs)
                    assert len(pred_objs_feats.size()) == 4
                    if loss == 'l1':
                        gap_loss_img = torch.norm((pred_objs_feats - target_objs_feats), p=1)
                        gap_loss = gap_loss + gap_loss_img /(pred_objs_feats.size(1)*pred_objs_feats.size(2) * pred_objs_feats.size(3))
                    layers_feats = layers[1:]
                    gaps = gap[1:]
                else:
                    layers_feats = layers
                    gaps = gap
                gap_loss_layers = self._get_feature_loss_with_t(loc_teacher, pred_objs, target_objs, layers_feats, loss, gaps)
                gap_loss = gap_loss + gap_loss_layers 
        gap_loss = gap_loss / pred_objs_all.size(0)

        return gap_loss

    def _compute_feats_loss_in_level(self, pred_objs_feat, target_objs_feat, loss, gap):
        """
        pred_objs_feat, target_objs_feat: feat from each level
        """
        if gap is None:
            pred_objs_feat = pred_objs_feat
            target_objs_feat = target_objs_feat
        else:
            pred_objs_feat = gap(pred_objs_feat)
            target_objs_feat = gap(target_objs_feat)
        assert len(pred_objs_feat.size()) == 4, len(target_objs_feat.size()) == 4
        if loss == 'l1':
            gap_loss_ =torch.norm((pred_objs_feat - target_objs_feat), p=1) 
            gap_loss_ = gap_loss_ /(pred_objs_feat.size(1)*pred_objs_feat.size(2) * pred_objs_feat.size(3))
        return gap_loss_

    def _get_feature_loss_with_t(self, loc_teacher, pred_objs, target_objs, layers, loss, gaps):
        # fn = loc_teacher.module.get_features #if is_parallel(loc_teacher) else loc_teacher.get_features
        fn = loc_teacher.get_features
        pred_objs_feats = fn(pred_objs, layers)
        with torch.no_grad(): #need grad to input so that influence the pred_bbox
            target_objs_feats = fn(target_objs, layers)
        gap_loss_list_ = [self._compute_feats_loss_in_level(pred_objs_feats[i], target_objs_feats[i], loss, gaps[i]) for i in range(len(layers))]
        gap_loss_list = gap_loss_list_
        gap_loss = sum(gap_loss_list) / len(gap_loss_list)
        return gap_loss
    
        
    def forward(self, pos_bbox_pred, pos_bbox_targets, img, loc_teacher, sum_pos=None, pos_bbox_labels=None, loss=None, gap=None, gap_size=None, layer=None):
        # import ipdb; ipdb.set_trace()
        if gap[0] is None or isinstance(gap[0], nn.AdaptiveAvgPool2d):
            if sum_pos:
                distill_loc_loss = self._forward_pool(pos_bbox_pred, pos_bbox_targets, img, loc_teacher, pos_bbox_labels=pos_bbox_labels, loss=loss, gap=gap, layer=layer)

        return distill_loc_loss

    