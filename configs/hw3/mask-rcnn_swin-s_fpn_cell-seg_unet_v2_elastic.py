_base_ = './mask-rcnn_swin-s_fpn_cell-seg.py'

# U-Net style enhancement V2: elastic deformation augmentation
# Requires: pip install albumentations
backend_args = None

albu_train_transforms = [
    dict(type='ElasticTransform', alpha=120, sigma=7, alpha_affine=7, p=0.35)
]

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=backend_args,
        imdecode_backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        skip_img_without_anno=True),
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

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
work_dir = './work_dirs/hw3_cell_seg_unet_v2_elastic'
test_evaluator = dict(outfile_prefix='./work_dirs/hw3_cell_seg_unet_v2_elastic/test_results')
