# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss


def kl_div(ps, qt):
    eps = 1e-10
    ps = ps + eps
    qt = qt + eps
    loss = qt * torch.log(qt) - qt * torch.log(ps)
    return loss.sum(1)


def kl_bce(ps, qt, label):
    # ps, qt: N*80 ==> N*80*2 for [0, 1]
    eps = 1e-10

    ps_1 = 1.0 - ps #N*80
    ps = torch.stack((ps, ps_1), dim=2)

    qt_1 = 1.0 - qt
    qt = torch.stack((qt, qt_1), dim=2)

    n_s = list(range(0, label.size(0), 1))
    ps[n_s,label] = 1.0 - ps[n_s,label] # N*80*2
    qt[n_s,label] = 1.0 - qt[n_s,label] # N*80*2

    ps = torch.clamp(ps, min=eps)
    qt = torch.clamp(qt, min=eps)
    # compute kl alond dim=2
    loss = qt * torch.log(qt) - qt * torch.log(ps)
    loss = loss.sum(2).sum(1) / ps.size(1)

    return loss


@LOSSES.register_module()
class KLDivergenceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 T=1.0,
                 ltype=None):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(KLDivergenceLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.T = T
        self.type = ltype

    def forward(self,
                cls_score,
                label,
                logits,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                T=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): the labels.
            logits (torch.tensor): the logits from cls_teachers
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        if T is None:
            self.T = self.T
        else:
            self.T = T
        # import ipdb; ipdb.set_trace()
        if self.use_sigmoid:
            p_s = torch.sigmoid(cls_score/self.T)
            p_t = torch.sigmoid(logits/self.T)
            if self.type == 'kl-ce':
                p_s = F.normalize(p_s, dim=1, p=1)
                p_t = F.normalize(p_t, dim=1, p=1)
                loss = kl_div(p_s, p_t)
            elif self.type == 'kl-bce':
                loss = kl_bce(p_s, p_t, label)
        else:
            p_s = F.softmax(cls_score/self.T, dim=1)
            p_t = F.softmax(logits/self.T, dim=1)
            if cls_score.size(1) != logits.size(1):
                p_t_gt = torch.zeros(p_t.size(0),1).to(p_t.device)
                p_t = torch.cat([p_t, p_t_gt], dim=1)
            if self.type == 'kl-ce':
                 loss = kl_div(p_s, p_t)

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        
        if avg_factor == 0:
            loss = torch.tensor(0.).to(cls_score.device)
        else:
            loss = weight_reduce_loss(
                loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        loss_kt_cls = self.loss_weight * self.T * self.T * loss
        return loss_kt_cls