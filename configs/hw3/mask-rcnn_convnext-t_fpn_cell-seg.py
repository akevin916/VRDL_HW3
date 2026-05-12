_base_ = '../convnext/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py'

# 4-class cell instance segmentation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=4),
        mask_head=dict(num_classes=4)),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.01,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=300,
            mask_thr_binary=0.5)))

# Dataset
dataset_type = 'CocoDataset'
data_root = 'data/'
backend_args = None
metainfo = dict(classes=('class1', 'class2', 'class3', 'class4'))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=backend_args,
        imdecode_backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 1333), (500, 1333), (600, 1333)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=backend_args,
        imdecode_backend='pillow'),
    dict(type='Resize', scale=(800, 1333), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='train/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test_image_info.json',
        data_prefix=dict(img='test_release/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test_image_info.json',
    metric=['bbox', 'segm'],
    format_only=True,
    outfile_prefix='./work_dirs/hw3_convnext_t/test_results',
    backend_args=backend_args)

# Schedule / optimizer
max_epochs = 100
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[80, 95],
        gamma=0.1),
]

optim_wrapper = dict(
    optimizer=dict(lr=5e-5),
    clip_grad=dict(max_norm=1.0, norm_type=2))

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='coco/segm_mAP_50',
        rule='greater',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

auto_scale_lr = dict(enable=False, base_batch_size=16)

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer')

work_dir = './work_dirs/hw3_convnext_t'
