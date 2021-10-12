# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
import inspect

import torch

from mmcv.utils import Registry, build_from_cfg

from mmcv.runner.optimizer.default_constructor import DefaultOptimizerConstructor
from mmcv.runner.optimizer.builder import register_torch_optimizers, build_optimizer_constructor, OPTIMIZERS, OPTIMIZER_BUILDERS


def build_optimizer_loc(model, cfg):
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = build_optimizer_constructor(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    optimizer = customize_call(optim_constructor, model)
    return optimizer

def customize_call(optim_constructor, model):
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optim_constructor.optimizer_cfg.copy()
    # if no paramwise option is specified, just use the global setting
    if not optim_constructor.paramwise_cfg:
        optim_params = []
        for k, p in model.named_parameters():
            if not k.startswith('loc_teacher'):
               optim_params.append(p)       
        optimizer_cfg['params'] = optim_params

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)