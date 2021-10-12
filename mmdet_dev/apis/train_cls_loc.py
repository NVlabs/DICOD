# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner)
from mmcv_dev.runner.optimizer import build_optimizer_loc
from mmcv.utils import build_from_cfg
from mmcv_dev.runner import EpochBasedRunnerFeatClsLoc


from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger

from mmcv.runner import load_checkpoint
from mmcv import Config
import torch.nn as nn
import copy
from cls_libs.cls_utils.models import ResNet_cls
from mmdet.models import build_detector


import torchvision.models as models


## define and load teacher models 
def build_cls_teacher(cfg_t, logger):
    """
    build and load our classification teacher model according to configurations of treacher in cfg
    """
    
    logger.info(f"=> Creating classifier teacher {cfg_t.arch}")
    backbone = models.__dict__[cfg_t.arch]()
    if cfg_t.arch.startswith('alexnet') or cfg_t.arch.startswith('vgg'):
        backbone.classifier[6] = nn.Linear(backbone.classifier[6].in_features, 80)
    elif cfg_t.arch.startswith('resnet') or cfg_t.arch.startswith('resnext'):
        backbone.fc = nn.Linear(backbone.fc.in_features, 80)

    if cfg_t.teacher_type == 'ssd':
        checkpoint = f"cls_teachers/ssd/model_best.pth.tar" 
    else:
        checkpoint = f"cls_teachers/general/model_best.pth.tar" 
    
    logger.info(f'=> ---- Loading classifier teacher: {checkpoint}')
    # import ipdb; ipdb.set_trace()
    load_checkpoint(backbone,
                    checkpoint,
                    map_location='cpu')
    teacher = ResNet_cls(backbone)

    return teacher


def train_detector_cls_loc(model,
                                dataset,
                                cfg,
                                distributed=False,
                                validate=False,
                                timestamp=None,
                                meta=None):
    logger = get_root_logger(cfg.log_level)
    if cfg.kt_config is not None and cfg.kt_config.cfg_t is not None:
        logger.info('===> Building and loading classifier teacher')
        teacher_cls = build_cls_teacher(cfg.kt_config.cfg_t, logger)
       
    else:
        logger.info('===> Training without classifier teacher')
        teacher_cls = None

  
    logger.info('===> Training without detector teacher')
    teacher_det = None # detection teacher is not used in this demo
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]
    if cfg.kt_config is not None:
        if cfg.kt_config.kt_loc is not None:
            logger.info('==> Build stn')
            model.build_stn() 
            model.kt_loc_turn_on = True
            logger.info(f'==> Build gap for kt loc: {model.gap}')
            if cfg.kt_config.kt_loc.loc_t and teacher_cls is not None:
                logger.info('==> Build and Load loc teacher')
                model.load_loc_teacher(teacher_cls)
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        find_unused_parameters = True
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if teacher_cls is not None:
            teacher_cls = MMDistributedDataParallel(
                teacher_cls.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        if teacher_cls is not None:
            teacher_cls = MMDataParallel(
            teacher_cls.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
      
  
    if cfg.kt_config is not None:
        if cfg.kt_config.kt_loc is not None:
            if cfg.kt_config.kt_loc.loc_t:
                optimizer = build_optimizer_loc(model, cfg.optimizer)
            else:
                optimizer = build_optimizer(model, cfg.optimizer)
        else: 
            optimizer = build_optimizer(model, cfg.optimizer)
    else: 
        optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
            teacher=[teacher_cls, teacher_det]))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        if isinstance(runner, EpochBasedRunner): # runner correspond to sampler
            runner.register_hook(DistSamplerSeedHook())
        
        if isinstance(runner, EpochBasedRunnerFeatClsLoc): # runner correspond to sampler
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
