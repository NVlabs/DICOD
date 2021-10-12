# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

_base_ = [
    '../../configs/_base_/models/ssd300.py', '../../configs/_base_/datasets/coco_detection.py',
    '../../configs/_base_/schedules/schedule_2x.py', '../../configs/_base_/default_runtime.py'
]
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/workspace/dataset/coco2017/' 
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(_delete_=True)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

runner = dict(type='EpochBasedRunnerFeatClsLoc', max_epochs=24)
kt_config = dict(
    cfg_t=dict(
        arch='resnet50',
        teacher_type='ssd',
        object_size=112,
    ),
    kt_cls=dict(
        type='KLDivergenceLoss', 
        T=2.0, 
        loss_weight=0.4, 
        ltype='kl-ce'

    ),
    kt_loc=dict(
        loc_t=True,
        type='l1', 
        gap='AdaptiveAvgPool',
        layer=[-1, 0],
        object_size=112,
        gap_size=4,
        loss_weight=0.15
    ),
    kt_feat=None
) #

model = dict(
        type='SingleStageDetectorFeatLoc', 
        bbox_head=dict(
                     type='SSDHeadClsLoc', 
                     loss_kt_cls=kt_config['kt_cls']
                     )
        ) 