_base_ = './mask-rcnn_swin-s_fpn_cell-seg.py'

# U-Net style enhancement V1: mirror padding (reflect)
backend_args = None

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
    dict(type='Pad', size_divisor=32, padding_mode='reflect'),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
work_dir = './work_dirs/hw3_cell_seg_unet_v1_mirrorpad'
test_evaluator = dict(outfile_prefix='./work_dirs/hw3_cell_seg_unet_v1_mirrorpad/test_results')

max_epochs = 50