_base_ = './mask-rcnn_swin-s_fpn_cell-seg.py'

# Use val set for parameter sweep evaluation.
test_dataloader = _base_.val_dataloader
test_evaluator = _base_.val_evaluator
