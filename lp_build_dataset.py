#!/usr/bin/env python3
"""
lp_build_dataset.py — 從 selected_plates_full.json 建立完整 YOLOv8-Pose 測試資料集
==================================================================================
用法：
    python lp_build_dataset.py --json selected_plates_429_full.json --output-dir ./lp_dataset/
    python lp_build_dataset.py --json selected_plates_429_full.json --output-dir ./lp_dataset/ --split 0.8

產出目錄結構 (符合 YOLOv8 標準)：
    lp_dataset/
      images/
        train/    ← 訓練圖片
        val/      ← 驗證圖片
      labels/
        train/    ← 訓練標記
        val/      ← 驗證標記
      dataset.yaml

YOLOv8-Pose label 格式 (每行):
    0 cx cy w h  x_TL y_TL 2  x_TR y_TR 2  x_BR y_BR 2  x_BL y_BL 2
"""

import os, sys, json, shutil, random, argparse
from pathlib import Path

PATH_MAP = {'/home/itemhsu': '/Users/miniaicar'}

def map_path(p):
    for src, dst in PATH_MAP.items():
        if p.startswith(src):
            return p.replace(src, dst, 1)
    return p


def quad_to_yolo(quad, img_w, img_h):
    """quad: [[x,y]x4] TL,TR,BR,BL → YOLO pose format string"""
    xs = [pt[0] for pt in quad]
    ys = [pt[1] for pt in quad]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    cx = max(0, min(1, (x0 + x1) / 2 / img_w))
    cy = max(0, min(1, (y0 + y1) / 2 / img_h))
    bw = max(0, min(1, (x1 - x0) / img_w))
    bh = max(0, min(1, (y1 - y0) / img_h))

    parts = [f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}']
    for pt in quad:
        kx = max(0, min(1, pt[0] / img_w))
        ky = max(0, min(1, pt[1] / img_h))
        parts.append(f'{kx:.6f} {ky:.6f} 2')

    return ' '.join(parts)


def build_dataset(json_path, output_dir, train_ratio=0.8, seed=42, use_corrected=True):
    print(f'Loading: {json_path}')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'  {len(data)} plate records')

    # 按原圖分組（一張圖可能有多個車牌）
    img_map = {}  # original_path → list of records
    for rec in data:
        orig = rec.get('original_path', '')
        if not orig:
            continue
        if orig not in img_map:
            img_map[orig] = []
        img_map[orig].append(rec)

    img_list = list(img_map.keys())
    print(f'  {len(img_list)} unique images')

    # 分割 train / val
    random.seed(seed)
    random.shuffle(img_list)
    n_train = int(len(img_list) * train_ratio)
    train_imgs = set(img_list[:n_train])
    val_imgs = set(img_list[n_train:])
    print(f'  Split: train={len(train_imgs)}, val={len(val_imgs)} (ratio={train_ratio})')

    # 建立目錄
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    stats = {'train_imgs': 0, 'val_imgs': 0, 'train_labels': 0, 'val_labels': 0,
             'skipped_no_img': 0, 'skipped_no_quad': 0}

    for orig_path, recs in img_map.items():
        local_path = map_path(orig_path)
        if not os.path.exists(local_path):
            stats['skipped_no_img'] += 1
            continue

        split = 'train' if orig_path in train_imgs else 'val'
        fname = os.path.basename(orig_path)
        fname_noext = os.path.splitext(fname)[0]

        # 複製圖片
        dst_img = os.path.join(output_dir, 'images', split, fname)
        if not os.path.exists(dst_img):
            shutil.copy2(local_path, dst_img)

        # 產生標記
        lines = []
        for rec in recs:
            quad = rec.get('corrected_quad') if use_corrected else rec.get('keypoints')
            if not quad or len(quad) != 4:
                quad = rec.get('keypoints')
            if not quad or len(quad) != 4:
                stats['skipped_no_quad'] += 1
                continue

            img_size = rec.get('image_size', [0, 0])
            img_h, img_w = img_size[0], img_size[1]
            if img_h == 0 or img_w == 0:
                continue

            lines.append(quad_to_yolo(quad, img_w, img_h))

        if lines:
            label_path = os.path.join(output_dir, 'labels', split, fname_noext + '.txt')
            with open(label_path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
            stats[f'{split}_labels'] += len(lines)

        stats[f'{split}_imgs'] += 1

    # dataset.yaml
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    abs_path = os.path.abspath(output_dir)
    yaml_content = f"""# YOLOv8-Pose License Plate Dataset
# Source: {os.path.basename(json_path)}
# Train/Val split: {train_ratio}/{1-train_ratio:.1f}

path: {abs_path}
train: images/train
val: images/val

kpt_shape: [4, 3]
# flip_idx: [1, 0, 3, 2]  # 水平翻轉時 TL↔TR, BL↔BR

names:
  0: license_plate

# Keypoints:
#   0: TL (top-left)
#   1: TR (top-right)
#   2: BR (bottom-right)
#   3: BL (bottom-left)
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f'\n✅ Dataset built: {abs_path}')
    print(f'   Train: {stats["train_imgs"]} images, {stats["train_labels"]} labels')
    print(f'   Val:   {stats["val_imgs"]} images, {stats["val_labels"]} labels')
    if stats['skipped_no_img']:
        print(f'   ⚠ Skipped (image not found): {stats["skipped_no_img"]}')
    if stats['skipped_no_quad']:
        print(f'   ⚠ Skipped (no quad): {stats["skipped_no_quad"]}')
    print(f'   dataset.yaml: {yaml_path}')

    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='建立 YOLOv8-Pose 車牌測試資料集')
    parser.add_argument('--json', type=str, default='selected_plates_429_full.json',
                        help='selected plates full JSON')
    parser.add_argument('--output-dir', type=str, default='./lp_dataset/',
                        help='輸出資料集目錄')
    parser.add_argument('--split', type=float, default=0.8,
                        help='train 比例 (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子')
    parser.add_argument('--use-original-kp', action='store_true',
                        help='使用原始 keypoints 而非 corrected_quad')
    args = parser.parse_args()

    build_dataset(
        args.json, args.output_dir,
        train_ratio=args.split,
        seed=args.seed,
        use_corrected=not args.use_original_kp,
    )
