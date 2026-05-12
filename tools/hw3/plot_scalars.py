import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def series(rows, key, x_key='step'):
    xs, ys = [], []
    for r in rows:
        if key in r and x_key in r:
            y = r[key]
            if isinstance(y, (int, float)):
                xs.append(r[x_key])
                ys.append(y)
    return xs, ys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scalars', required=True, help='Path to scalars.json')
    parser.add_argument('--out', required=True, help='Output png path')
    args = parser.parse_args()

    rows = load_jsonl(args.scalars)
    if not rows:
        raise RuntimeError('No valid rows found in scalars file.')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    # Training losses
    for k in ['loss', 'loss_cls', 'loss_bbox', 'loss_mask', 'loss_rpn_cls', 'loss_rpn_bbox']:
        x, y = series(rows, k)
        if x:
            axes[0].plot(x, y, label=k, linewidth=1.2)
    axes[0].set_title('Training Loss Curves')
    axes[0].set_xlabel('step')
    axes[0].set_ylabel('loss')
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=3)

    # Validation metrics (if available)
    has_metric = False
    for k in ['coco/segm_mAP_50', 'coco/segm_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP']:
        x, y = series(rows, k)
        if x:
            has_metric = True
            axes[1].plot(x, y, label=k, linewidth=1.5)

    if has_metric:
        axes[1].set_title('Validation Metrics')
        axes[1].set_xlabel('step')
        axes[1].set_ylabel('AP')
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=9)
    else:
        axes[1].text(0.5, 0.5, 'No validation AP metrics found in scalars.json',
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_axis_off()

    fig.suptitle('Swin-s Mask R-CNN Training Summary', fontsize=14)
    fig.savefig(out_path, dpi=150)
    print(f'Saved plot: {out_path}')


if __name__ == '__main__':
    main()
