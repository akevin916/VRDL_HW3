"""
快速驗證 COCO JSON 格式的細胞資料集是否正確。

功能：
  1. 統計 images / annotations / categories 數量
  2. 檢查 image_id 和 category_id 的一致性
  3. 隨機抽取 N 張圖，疊加 mask 輸出到 output_dir 以便目視確認

用法：
  # 只做統計檢查（不存圖）
  python tools/dataset_converters/verify_coco.py \
      --ann data/annotations/train.json \
      --img-dir data/train

  # 同時輸出視覺化圖片
  python tools/dataset_converters/verify_coco.py \
      --ann data/annotations/train.json \
      --img-dir data/train \
      --vis-dir /tmp/vis \
      --num-vis 5
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')            # 無 GUI 環境也能存圖
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba


COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
          '#911eb4', '#42d4f4', '#f032e6', '#bfef45']


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def stats(coco: dict):
    print('=' * 50)
    print(f"  categories : {len(coco['categories'])}")
    for c in coco['categories']:
        print(f"    {c['id']:2d}  {c['name']}")
    print(f"  images     : {len(coco['images'])}")
    print(f"  annotations: {len(coco['annotations'])}")

    # annotations per category
    cat_count = defaultdict(int)
    for ann in coco['annotations']:
        cat_count[ann['category_id']] += 1
    id_to_name = {c['id']: c['name'] for c in coco['categories']}
    print('\n  Annotations per class:')
    for cat_id in sorted(cat_count):
        print(f"    {id_to_name.get(cat_id, '?'):10s}: {cat_count[cat_id]}")

    # images with 0 annotations
    img_ids_with_ann = {a['image_id'] for a in coco['annotations']}
    all_img_ids = {img['id'] for img in coco['images']}
    empty_imgs = all_img_ids - img_ids_with_ann
    if empty_imgs:
        print(f'\n  ⚠️  Images with 0 annotations: {len(empty_imgs)}')
    else:
        print('\n  ✅ All images have at least 1 annotation')
    print('=' * 50)


def check_integrity(coco: dict):
    """檢查 image_id / category_id 是否都合法"""
    valid_img_ids = {img['id'] for img in coco['images']}
    valid_cat_ids = {c['id'] for c in coco['categories']}
    errors = []
    for ann in coco['annotations']:
        if ann['image_id'] not in valid_img_ids:
            errors.append(f"ann {ann['id']}: invalid image_id {ann['image_id']}")
        if ann['category_id'] not in valid_cat_ids:
            errors.append(f"ann {ann['id']}: invalid category_id {ann['category_id']}")
        if ann['area'] <= 0:
            errors.append(f"ann {ann['id']}: area={ann['area']} <= 0")
        if len(ann.get('segmentation', [])) == 0:
            errors.append(f"ann {ann['id']}: empty segmentation")
    if errors:
        print(f'\n  ❌ Found {len(errors)} integrity errors (showing first 10):')
        for e in errors[:10]:
            print(f'     {e}')
    else:
        print('  ✅ Integrity check passed')


def draw_sample(img_path: Path, anns: list, id_to_name: dict, out_path: Path):
    """讀取一張圖，疊加 polygon 遮罩並存檔"""
    try:
        img = tifffile.imread(str(img_path))
    except Exception as e:
        print(f'  Cannot read {img_path}: {e}')
        return

    # 統一轉成 3 channel uint8
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)

    legend_handles = {}
    for ann in anns:
        cat_id = ann['category_id']
        color = COLORS[(cat_id - 1) % len(COLORS)]
        rgba = to_rgba(color, alpha=0.4)

        # 畫 polygon
        for seg in ann['segmentation']:
            xs = seg[0::2]
            ys = seg[1::2]
            poly = mpatches.Polygon(
                list(zip(xs, ys)),
                closed=True,
                facecolor=rgba,
                edgecolor=color,
                linewidth=1)
            ax.add_patch(poly)

        # 畫 bbox
        x, y, w, h = ann['bbox']
        rect = mpatches.Rectangle(
            (x, y), w, h,
            linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        name = id_to_name.get(cat_id, str(cat_id))
        if cat_id not in legend_handles:
            legend_handles[cat_id] = mpatches.Patch(color=color, label=name)

    ax.legend(handles=list(legend_handles.values()),
               loc='upper right', fontsize=8)
    ax.set_title(img_path.parent.name if img_path.parent.name != '.' else img_path.name,
                 fontsize=9)
    ax.axis('off')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches='tight', dpi=100)
    plt.close(fig)
    print(f'  Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', required=True,
                        help='COCO JSON 路徑，例如 data/annotations/train.json')
    parser.add_argument('--img-dir', required=True,
                        help='影像根目錄，例如 data/train')
    parser.add_argument('--vis-dir', default=None,
                        help='視覺化輸出目錄（不指定則跳過出圖）')
    parser.add_argument('--num-vis', type=int, default=5,
                        help='隨機抽取幾張圖做視覺化（預設 5）')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print(f'\n載入：{args.ann}')
    coco = load_json(args.ann)

    stats(coco)
    check_integrity(coco)

    if args.vis_dir is None:
        return

    # ── 視覺化 ──────────────────────────────────────────
    random.seed(args.seed)
    img_dir = Path(args.img_dir)
    vis_dir = Path(args.vis_dir)

    # 建立 image_id → annotations 索引
    id_to_anns = defaultdict(list)
    for ann in coco['annotations']:
        id_to_anns[ann['image_id']].append(ann)

    id_to_name = {c['id']: c['name'] for c in coco['categories']}

    # 只抽有 annotation 的圖
    imgs_with_ann = [img for img in coco['images']
                     if id_to_anns[img['id']]]
    samples = random.sample(imgs_with_ann, min(args.num_vis, len(imgs_with_ann)))

    print(f'\n輸出 {len(samples)} 張視覺化圖 → {vis_dir}')
    for img_info in samples:
        # file_name 可能是 "<uuid>/image.tif" 或 "<uuid>.tif"
        img_path = img_dir / img_info['file_name']
        anns = id_to_anns[img_info['id']]
        stem = Path(img_info['file_name']).parent.name or Path(img_info['file_name']).stem
        out_path = vis_dir / f'{stem}.png'
        draw_sample(img_path, anns, id_to_name, out_path)


if __name__ == '__main__':
    main()
