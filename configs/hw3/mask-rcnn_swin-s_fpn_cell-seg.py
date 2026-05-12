# ============================================================
# Homework 3 — Cell Instance Segmentation (4 classes)
# Architecture : Mask R-CNN + Swin-S + FPN
# Backbone     : Swin-S  (ImageNet-1K pretrained, ~69M params)
# Dataset      : 209 train / 101 test .tif images, COCO format
# Target metric: AP50 (instance segmentation)
#
# 繼承鏈（由下往上）：
#   mask-rcnn_r50_fpn.py              ← 完整模型骨架（ResNet50 預設）
#   coco_instance.py                  ← 資料集預設
#   schedule_1x.py                    ← 12 epoch 排程預設
#   mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py
#     ← 換掉 backbone 為 Swin-T、啟用多尺度 crop、AdamW、36 epoch
#   mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco.py
#     ← 加上 AmpOptimWrapper（混合精度）
#   mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py  ← 換成 Swin-S 深度
#
# 本 config 只需要覆蓋：
#   ① num_classes 80 → 4
#   ② 資料集路徑 + metainfo
#   ③ max_epochs 36 → 100、LR 調整
# ============================================================

_base_ = '../swin/mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py'

# ─────────────────────────────────────────────
# 1. Model：只改類別數與 drop_path（其餘全繼承）
# ─────────────────────────────────────────────
model = dict(
    backbone=dict(
        drop_path_rate=0.3),           # 預設 0.2，小資料集稍微加強正則化
    roi_head=dict(
        bbox_head=dict(num_classes=4),
        mask_head=dict(num_classes=4)),
    # ── Test config: low score_thr for AP50 ───
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.01,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=300,           # val sweep best
            mask_thr_binary=0.5)))

# ─────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────
# Expected directory structure:
#   data/
#     train/                  ← 209 .tif training images
#     test_release/           ← 101 .tif test images
#     annotations/
#       train.json            ← COCO-format train annotations
#       val.json              ← COCO-format val annotations (split from train)

dataset_type = 'CocoDataset'
data_root = 'data/'
backend_args = None

# ── 4 cell class names ──────
metainfo = dict(classes=('class1', 'class2', 'class3', 'class4'))

# ── train_pipeline：在繼承的 ms-crop 基礎上加垂直翻轉 ──
# base 只有 RandomFlip(prob=0.5, direction='horizontal')
# 細胞無方向性，加上 vertical 翻轉可提升資料多樣性
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
        data_prefix=dict(img='train/'),   # val images also reside in train/
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

# ── Test dataloader (no GT, for competition submission) ──
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

# ── Evaluators ────────────────────────────────
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)

# Test evaluator: format_only outputs predictions for submission
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test_image_info.json',
    metric=['bbox', 'segm'],
    format_only=True,
    outfile_prefix='./work_dirs/hw3_cell_seg/test_results',
    backend_args=backend_args)

# ─────────────────────────────────────────────
# 3. Training schedule
# ─────────────────────────────────────────────
max_epochs = 100

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    # Linear warmup for first 500 iters (small dataset = fewer iters/epoch)
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    # Cosine-style step decay: drop LR at 80% and 95% of total epochs
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[80, 95],
        gamma=0.1),
]

# ── Optimizer：繼承 AdamW + paramwise_cfg，只改 lr 並加 clip_grad ──
optim_wrapper = dict(
    optimizer=dict(lr=5e-5),                     # 預設 1e-4，小資料集調低
    clip_grad=dict(max_norm=1.0, norm_type=2))   # gradient clip 增加穩定性

# ─────────────────────────────────────────────
# 4. Runtime
# ─────────────────────────────────────────────
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,                  # save every 5 epochs
        save_best='coco/segm_mAP_50', # keep best mask AP50 checkpoint
        rule='greater',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# Auto-scale LR when changing batch size (base: 16 images total = 8 GPU x 2)
auto_scale_lr = dict(enable=False, base_batch_size=16)

# Visualization backends
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer')

# Work directory
work_dir = './work_dirs/hw3_cell_seg'
