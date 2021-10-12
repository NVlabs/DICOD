# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os.path as osp
import platform
import shutil
import time
import warnings

import torch
from torchvision.utils import save_image 

import mmcv
from .base_runner_feat_cls_loc import BaseRunnerFeatClsLoc
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.utils import get_host_info
import torch.nn.functional as F


def get_info_t(data_batch, teacher, cfg):
    """
    get the logits from the classification teacher for KD_cls 
    """
    teacher.eval()
    boxes_list = data_batch['gt_bboxes'].data[0]
    images =  data_batch['img'].data[0]
    height, width = images.size(2), images.size(3)
    nobj = [0]
    img_cls = []

    for (ni, img) in enumerate(images):
        input_img= img.unsqueeze(0).cuda()
        box_list =  boxes_list[ni].cuda()  
        nobj.append(nobj[ni]+box_list.size(0))  
        if box_list.size(0) < 1:
            continue
        left = torch.clamp(box_list[ :, 0], min=0)
        bottom = torch.clamp(box_list[:, 1], min=0)
        right = torch.clamp(box_list[:, 2], max=width)
        top = torch.clamp(box_list[:, 3], max=height)

        theta = torch.zeros((box_list.size(0), 2, 3), dtype=torch.float).cuda()
        theta[:, 0, 0] = (right-left)/width
        theta[:, 1, 1] = (top-bottom)/height
        
        theta[:, 0, 2] =  -1. +  (left + (right-left)/2)/(width/2)
        theta[:, 1, 2] =  -1. + ((bottom + (top-bottom)/2))/(height/2)
        
        grid_size = torch.Size((box_list.size(0), input_img.size(1), cfg.object_size, cfg.object_size))
        grid = F.affine_grid(theta, grid_size, align_corners=False) 
        img_ = F.grid_sample(torch.cat(box_list.size(0)*[input_img]), grid, align_corners=False) 
        img_cls.append(img_)

    imgs_cls = torch.cat(img_cls, dim=0)
    with torch.no_grad():
        logit_t_ = teacher(imgs_cls).cpu()
      
    for i in range(images.size(0)):     
        if nobj[i] == nobj[i+1]:
            logits_t = []  ##  something for no objects
        else:
            logits_t = [logit_t_[nobj[i]:nobj[i+1]]]
        data_batch['img_metas'].data[0][i].update({'gt_logits_t': logits_t})

    return data_batch 

@RUNNERS.register_module()
class EpochBasedRunnerFeatClsLoc(BaseRunnerFeatClsLoc):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """
    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        if self.teacher_det is not None:
            self.teacher_det.train()
        if self.teacher_cls is not None:
            self.teacher_cls.eval()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)

        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            # update data_batch  
            if self.batch_processor is None:
                if self.teacher_det is not None:
                    with torch.no_grad():
                        t_info = self.teacher_det.train_step(data_batch, self.optimizer, epoch=self.epoch,
                                        iter=self._inner_iter, teach=True, t_info=None,
                                        **kwargs)
                else:
                    t_info = None
                if self.teacher_cls is not None:
                    data_batch = get_info_t(data_batch,  self.teacher_cls, self.model.module.kt_cfg.cfg_t)

                outputs = self.model.train_step(data_batch, self.optimizer, epoch=self.epoch,
                                      iter=self._inner_iter, teach=False, t_info=t_info,
                                      **kwargs)
            else:
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=True, **kwargs)

            if not isinstance(outputs, dict):
                raise TypeError('"batch_processor()" or "model.train_step()"'
                                ' must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1
    
    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                # import ipdb; ipdb.set_trace()
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

@RUNNERS.register_module()
class RunnerFeatClsLoc(EpochBasedRunnerFeatClsLoc):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead')
        super().__init__(*args, **kwargs)
# 