# Homework 3 Environment Setup

## 1) Create a clean conda environment

```bash
conda create -n mmd python=3.10 -y
conda activate mmd
python --version
```

## 2) Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## 3) Install MMEngine / MMCV

```bash
pip install -U openmim
mim install "mmengine>=0.7.1,<1.0.0"
mim install "mmcv>=2.0.0,<2.2.0"
```

## 4) Install dependencies

```bash
cd /home/cvml_7/Desktop/2026_class/mmdetection
pip install -r requirements.txt
pip install tifffile scikit-image imagecodecs pyyaml tensorboard
```

## 5) Install MMDetection (development mode)

```bash
pip install -U "setuptools>=64"
pip install -v -e . --no-build-isolation
```

## 6) Verify installation

```bash
python -c "import torch, mmcv, mmengine, mmdet; print('torch', torch.__version__); print('mmcv', mmcv.__version__); print('mmengine', mmengine.__version__); print('mmdet', mmdet.__version__)"
```

## 7) (Optional) Data conversion and verification

```bash
python tools/dataset_converters/cell2coco.py --data-root data/ --val-ratio 0.2 --seed 42
python tools/dataset_converters/verify_coco.py --ann data/annotations/train.json --img-dir data/train --vis-dir /tmp/cell_vis --num-vis 5
```

# Training

```bash
python tools/train.py configs/hw3/mask-rcnn_swin-s_fpn_cell-seg.py
```

## 1) TensorBoard monitoring

```bash
tensorboard --logdir work_dirs/hw3_cell_seg --port 6006
```

Open in browser:

`http://localhost:6006`

## 2) Alternative model training

Two alternative models for fair comparison with the Swin-S baseline.

### A. ConvNeXt-T + Mask R-CNN

Config file:

`configs/hw3/mask-rcnn_convnext-t_fpn_cell-seg.py`

Install required package (ConvNeXt backbone requires mmpretrain):

```bash
pip install mmpretrain
```

Train:

```bash
python tools/train.py configs/hw3/mask-rcnn_convnext-t_fpn_cell-seg.py
```

Test and generate submission file:

```bash
python tools/test.py configs/hw3/mask-rcnn_convnext-t_fpn_cell-seg.py \
  work_dirs/hw3_convnext_t/best_coco_segm_mAP_50_epoch_45.pth
```

### B. HTC-R50 (without semantic)

Note: HTC's `bbox_head` / `mask_head` is a 3-stage list, so the number of classes must be updated in all 3 stages.

Config file:

`configs/hw3/htc_r50_fpn_cell-seg.py`

Train:

```bash
python tools/train.py configs/hw3/htc_r50_fpn_cell-seg.py
```

Test and generate submission file:

```bash
python tools/test.py configs/hw3/htc_r50_fpn_cell-seg.py \
  work_dirs/hw3_htc_r50/best_coco_segm_mAP_50_epoch_XX.pth
```

## 3) U-Net inspired augmentation variants (Mask R-CNN)

1. Mirror padding (reflect boundary padding)
   [configs/hw3/mask-rcnn_swin-s_fpn_cell-seg_unet_v1_mirrorpad.py](configs/hw3/mask-rcnn_swin-s_fpn_cell-seg_unet_v1_mirrorpad.py)
2. Elastic deformation augmentation
   [configs/hw3/mask-rcnn_swin-s_fpn_cell-seg_unet_v2_elastic.py](configs/hw3/mask-rcnn_swin-s_fpn_cell-seg_unet_v2_elastic.py)
3. Weighted loss interface stub (loss modification not yet enabled)
   [configs/hw3/mask-rcnn_swin-s_fpn_cell-seg_unet_v3_lossstub.py](configs/hw3/mask-rcnn_swin-s_fpn_cell-seg_unet_v3_lossstub.py)

### V1 Train / Test

```bash
python tools/train.py configs/hw3/mask-rcnn_swin-s_fpn_cell-seg_unet_v1_mirrorpad.py
python tools/test.py configs/hw3/mask-rcnn_swin-s_fpn_cell-seg_unet_v1_mirrorpad.py \
  work_dirs/hw3_cell_seg_unet_v1_mirrorpad/best_coco_segm_mAP_50_epoch_XX.pth
```

### V2 Train / Test

```bash
# Install if not already installed
pip install albumentations

python tools/train.py configs/hw3/mask-rcnn_swin-s_fpn_cell-seg_unet_v2_elastic.py
python tools/test.py configs/hw3/mask-rcnn_swin-s_fpn_cell-seg_unet_v2_elastic.py \
  work_dirs/hw3_cell_seg_unet_v2_elastic/best_coco_segm_mAP_50_epoch_XX.pth
```

### V3 Train / Test (loss unchanged)

```bash
python tools/train.py configs/hw3/mask-rcnn_swin-s_fpn_cell-seg_unet_v3_lossstub.py
python tools/test.py configs/hw3/mask-rcnn_swin-s_fpn_cell-seg_unet_v3_lossstub.py \
  work_dirs/hw3_cell_seg_unet_v3_lossstub/best_coco_segm_mAP_50_epoch_XX.pth
```


# Performance Snapshot
![alt text](resources/snapshot.png)
