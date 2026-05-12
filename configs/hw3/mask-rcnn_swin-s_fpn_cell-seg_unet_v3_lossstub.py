_base_ = './mask-rcnn_swin-s_fpn_cell-seg.py'

# U-Net style enhancement V3: weighted-loss interface placeholder
# NOTE:
#   The U-Net boundary-weighted pixel loss is not directly compatible with
#   Mask R-CNN/HTC instance mask training flow.
#   This config intentionally keeps the original loss unchanged for fair test.
#   Use this as the dedicated branch to implement custom loss in code later.

work_dir = './work_dirs/hw3_cell_seg_unet_v3_lossstub'
test_evaluator = dict(outfile_prefix='./work_dirs/hw3_cell_seg_unet_v3_lossstub/test_results')
