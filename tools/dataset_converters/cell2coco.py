"""
Convert cell segmentation dataset to COCO instance segmentation format.

資料夾結構：
  data/
    train/
      <uuid>/
        image.tif          ← 原始影像
        class1.tif         ← 實例遮罩，每個不同非零值 = 一個細胞實例
        class2.tif         ← 同上
    test_release/
      <uuid>.tif
    test_image_name_to_ids.json

輸出：
  data/annotations/train.json
  data/annotations/val.json
  data/annotations/test_image_info.json  ← test 用（無 GT）

用法：
  python tools/dataset_converters/cell2coco.py \
      --data-root data/ \
      --val-ratio 0.2 \
      --seed 42
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tifffile
from skimage import measure


# ── COCO 類別定義（index 從 1 開始）─────────────────────
CATEGORIES = [
    {'id': 1, 'name': 'class1', 'supercategory': 'cell'},
    {'id': 2, 'name': 'class2', 'supercategory': 'cell'},
    {'id': 3, 'name': 'class3', 'supercategory': 'cell'},
    {'id': 4, 'name': 'class4', 'supercategory': 'cell'},
]
CLASS_NAME_TO_ID = {c['name']: c['id'] for c in CATEGORIES}


def mask_to_polygons(binary_mask: np.ndarray) -> list[list[float]]:
    """
    將單一實例的二值遮罩轉換為 COCO polygon 格式。
    回傳 list of [x1,y1,x2,y2,...] 多邊形（可能有多個輪廓）。
    面積太小（< 1 pixel）的輪廓會被過濾。
    """
    polygons = []
    # skimage.measure.find_contours 回傳 (row, col) 格式
    contours = measure.find_contours(binary_mask.astype(np.uint8), 0.5)
    for contour in contours:
        # 轉成 [x, y, x, y, ...] 並攤平
        contour = np.flip(contour, axis=1)  # (row,col) → (x,y)
        seg = contour.flatten().tolist()
        if len(seg) >= 6:  # 至少 3 個點才是合法多邊形
            polygons.append(seg)
    return polygons


def get_bbox_from_mask(binary_mask: np.ndarray) -> list[float]:
    """
    從二值遮罩計算 COCO bbox：[x_min, y_min, width, height]
    """
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [
        float(x_min),
        float(y_min),
        float(x_max - x_min + 1),
        float(y_max - y_min + 1),
    ]


def process_sample(
    sample_dir: Path,
    image_id: int,
    ann_id_start: int,
) -> tuple[dict, list[dict]]:
    """
    處理一個訓練樣本資料夾，回傳 (image_info, annotations)。
    """
    img_path = sample_dir / 'image.tif'
    img = tifffile.imread(str(img_path))

    # 取得影像高寬（支援灰階、RGB、多通道）
    if img.ndim == 2:
        height, width = img.shape
    else:
        height, width = img.shape[:2]

    image_info = {
        'id': image_id,
        'file_name': sample_dir.name + '/image.tif',  # 相對於 data/train/
        'height': height,
        'width': width,
    }

    annotations = []
    ann_id = ann_id_start

    # 掃描所有 classX.tif（X = 1~4）
    for class_name, cat_id in CLASS_NAME_TO_ID.items():
        mask_path = sample_dir / f'{class_name}.tif'
        if not mask_path.exists():
            continue

        labeled_mask = tifffile.imread(str(mask_path))

        # 若是 float 遮罩，轉成 uint
        if labeled_mask.dtype.kind == 'f':
            labeled_mask = labeled_mask.astype(np.int32)

        # 若是二值遮罩（只有 0/1），用 connected components 找各實例
        unique_vals = np.unique(labeled_mask)
        unique_vals = unique_vals[unique_vals != 0]  # 排除背景

        if len(unique_vals) == 1 and unique_vals[0] == 1:
            # 可能是二值遮罩：用 connected components 分割實例
            labeled_mask = measure.label(labeled_mask)
            unique_vals = np.unique(labeled_mask)
            unique_vals = unique_vals[unique_vals != 0]

        for instance_val in unique_vals:
            binary_mask = (labeled_mask == instance_val).astype(np.uint8)
            area = int(binary_mask.sum())

            if area < 1:
                continue

            polygons = mask_to_polygons(binary_mask)
            if not polygons:
                continue

            bbox = get_bbox_from_mask(binary_mask)

            annotations.append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': cat_id,
                'segmentation': polygons,
                'area': area,
                'bbox': bbox,
                'iscrowd': 0,
            })
            ann_id += 1

    return image_info, annotations


def build_coco_json(
    image_infos: list[dict],
    annotations: list[dict],
) -> dict:
    return {
        'info': {'description': 'Cell Instance Segmentation - HW3'},
        'licenses': [],
        'categories': CATEGORIES,
        'images': image_infos,
        'annotations': annotations,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert cell TIF dataset to COCO JSON format.')
    parser.add_argument('--data-root', default='data/',
                        help='根目錄（包含 train/ 和 test_release/）')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='驗證集比例（預設 0.2 = 20%%）')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子，確保可重現')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_dir = data_root / 'train'
    ann_dir = data_root / 'annotations'
    ann_dir.mkdir(exist_ok=True)

    # ── 掃描所有訓練樣本 ────────────────────────────────
    sample_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    print(f'找到 {len(sample_dirs)} 個訓練樣本')

    # ── 處理所有樣本 ─────────────────────────────────────
    all_image_infos = []
    all_annotations = []
    ann_id_counter = 1

    for img_id, sample_dir in enumerate(sample_dirs, start=1):
        image_info, anns = process_sample(sample_dir, img_id, ann_id_counter)
        all_image_infos.append(image_info)
        all_annotations.extend(anns)
        ann_id_counter += len(anns)
        print(f'  [{img_id:3d}/{len(sample_dirs)}] {sample_dir.name} '
              f'→ {len(anns)} annotations')

    print(f'\n總 annotation 數：{len(all_annotations)}')

    # ── Train / Val 分割 ──────────────────────────────────
    random.seed(args.seed)
    indices = list(range(len(all_image_infos)))
    random.shuffle(indices)

    n_val = max(1, int(len(indices) * args.val_ratio))
    val_indices = set(indices[:n_val])
    train_indices = set(indices[n_val:])

    train_images = [all_image_infos[i] for i in sorted(train_indices)]
    val_images   = [all_image_infos[i] for i in sorted(val_indices)]

    train_img_ids = {img['id'] for img in train_images}
    val_img_ids   = {img['id'] for img in val_images}

    train_anns = [a for a in all_annotations if a['image_id'] in train_img_ids]
    val_anns   = [a for a in all_annotations if a['image_id'] in val_img_ids]

    print(f'\nTrain: {len(train_images)} 張圖 / {len(train_anns)} annotations')
    print(f'Val  : {len(val_images)} 張圖 / {len(val_anns)} annotations')

    # ── 儲存 train.json ────────────────────────────────────
    train_json_path = ann_dir / 'train.json'
    with open(train_json_path, 'w') as f:
        json.dump(build_coco_json(train_images, train_anns), f)
    print(f'\n已儲存：{train_json_path}')

    # ── 儲存 val.json ─────────────────────────────────────
    val_json_path = ann_dir / 'val.json'
    with open(val_json_path, 'w') as f:
        json.dump(build_coco_json(val_images, val_anns), f)
    print(f'已儲存：{val_json_path}')

    # ── 產生 test_image_info.json（COCO 格式，無 annotations）──
    # 從現有的 test_image_name_to_ids.json 轉換
    test_id_file = data_root / 'test_image_name_to_ids.json'
    if test_id_file.exists():
        with open(test_id_file) as f:
            test_images_raw = json.load(f)
        test_json = {
            'info': {'description': 'Cell Instance Segmentation - HW3 Test'},
            'licenses': [],
            'categories': CATEGORIES,
            'images': test_images_raw,  # 已包含 id, file_name, height, width
            'annotations': [],
        }
        test_json_path = ann_dir / 'test_image_info.json'
        with open(test_json_path, 'w') as f:
            json.dump(test_json, f)
        print(f'已儲存：{test_json_path}')


if __name__ == '__main__':
    main()
