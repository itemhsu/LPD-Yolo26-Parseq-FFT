#!/usr/bin/env python3
"""
lp_merge_dataset.py — 合併自建車牌資料集到現有 Roboflow YOLOv8-Pose zip
====================================================================
用法：
    python lp_merge_dataset.py \
        --zip lp-det-v3-job3.v1i.yolov8.zip \
        --json selected_plates_429_full.json \
        --output merged_dataset.zip

功能：
    1. 解壓現有 zip（含 train/valid/test 三個 split）
    2. 從 selected_plates JSON 產生 YOLO label + 複製原圖
    3. 新增資料放入 train/（全部加到 train split）
    4. 更新 data.yaml
    5. 重新打包成新的 zip
"""

import os, sys, json, shutil, zipfile, argparse, re
from pathlib import Path

PATH_MAP = {'/home/itemhsu': '/Users/miniaicar'}

def map_path(p):
    for src, dst in PATH_MAP.items():
        if p.startswith(src):
            return p.replace(src, dst, 1)
    return p


def quad_to_yolo(quad, img_w, img_h):
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


def merge_dataset(zip_path, json_path, output_zip, target_split='train', use_corrected=True):
    work_dir = os.path.join(os.path.dirname(output_zip) or '.', '_merge_work')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    # Step 1: 解壓現有 zip
    print(f'[1] Extracting: {zip_path}')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(work_dir)

    # 確認目錄結構
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(work_dir, split, 'images')
        lbl_dir = os.path.join(work_dir, split, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        n_img = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        n_lbl = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
        print(f'    {split}: {n_img} images, {n_lbl} labels')

    # Step 2: 讀取 selected plates JSON
    print(f'[2] Loading: {json_path}')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'    {len(data)} plate records')

    # 按原圖分組
    img_map = {}
    for rec in data:
        orig = rec.get('original_path', '')
        if not orig:
            continue
        if orig not in img_map:
            img_map[orig] = []
        img_map[orig].append(rec)
    print(f'    {len(img_map)} unique images')

    # Step 3: 產生 label + 複製圖片到 target_split
    target_img_dir = os.path.join(work_dir, target_split, 'images')
    target_lbl_dir = os.path.join(work_dir, target_split, 'labels')

    added_img = 0
    added_lbl = 0
    skipped = 0

    for orig_path, recs in img_map.items():
        local_path = map_path(orig_path)
        if not os.path.exists(local_path):
            skipped += 1
            continue

        img_size = recs[0].get('image_size', [0, 0])
        img_h, img_w = img_size[0], img_size[1]
        if img_h == 0 or img_w == 0:
            skipped += 1
            continue

        # 產生不衝突的檔名：lp_加原始檔名
        fname = os.path.basename(orig_path)
        base, ext = os.path.splitext(fname)
        # 確保檔名唯一（加 lp_ prefix）
        safe_name = 'lp_' + re.sub(r'[^\w\-.]', '_', base)
        img_fname = safe_name + ext
        lbl_fname = safe_name + '.txt'

        dst_img = os.path.join(target_img_dir, img_fname)
        dst_lbl = os.path.join(target_lbl_dir, lbl_fname)

        # 跳過已存在的（避免重複合併）
        if os.path.exists(dst_img):
            continue

        # 複製圖片
        shutil.copy2(local_path, dst_img)
        added_img += 1

        # 產生 label
        lines = []
        for rec in recs:
            quad = rec.get('corrected_quad') if use_corrected else rec.get('keypoints')
            if not quad or len(quad) != 4:
                quad = rec.get('keypoints')
            if not quad or len(quad) != 4:
                continue
            lines.append(quad_to_yolo(quad, img_w, img_h))

        if lines:
            with open(dst_lbl, 'w') as f:
                f.write('\n'.join(lines) + '\n')
            added_lbl += len(lines)

    print(f'    Added: {added_img} images, {added_lbl} labels to {target_split}/')
    if skipped:
        print(f'    Skipped (image not found): {skipped}')

    # Step 4: 更新 data.yaml
    yaml_path = os.path.join(work_dir, 'data.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            original_yaml = f.read()
        print(f'[3] Original data.yaml preserved')
        # 追加註解
        with open(yaml_path, 'a') as f:
            f.write(f'\n# Merged {added_img} images from {os.path.basename(json_path)}\n')
    else:
        # 建立新的 data.yaml
        yaml_content = f"""path: .
train: train/images
val: valid/images
test: test/images

kpt_shape: [4, 3]

names:
  0: license_plate

# Merged {added_img} images from {os.path.basename(json_path)}
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

    # 最終統計
    print(f'[4] Final counts:')
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(work_dir, split, 'images')
        lbl_dir = os.path.join(work_dir, split, 'labels')
        n_img = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        n_lbl = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
        print(f'    {split}: {n_img} images, {n_lbl} labels')

    # Step 5: 打包成新 zip
    print(f'[5] Creating: {output_zip}')
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(work_dir):
            for f in sorted(files):
                fpath = os.path.join(root, f)
                arcname = os.path.relpath(fpath, work_dir)
                zf.write(fpath, arcname)

    zip_size = os.path.getsize(output_zip)
    print(f'    Size: {zip_size / 1024 / 1024:.1f} MB')

    # 清理
    shutil.rmtree(work_dir)
    print(f'\n✅ Done! Merged dataset: {output_zip}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='合併車牌資料集到 Roboflow YOLOv8 zip')
    parser.add_argument('--zip', type=str, required=True,
                        help='現有 Roboflow zip (lp-det-v3-job3.v1i.yolov8.zip)')
    parser.add_argument('--json', type=str, required=True,
                        help='selected_plates_429_full.json')
    parser.add_argument('--output', type=str, default='merged_dataset.zip',
                        help='輸出合併後的 zip')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'valid', 'test'],
                        help='新資料放入哪個 split (default: train)')
    parser.add_argument('--use-original-kp', action='store_true',
                        help='使用原始 keypoints 而非 corrected_quad')
    args = parser.parse_args()

    merge_dataset(
        args.zip, args.json, args.output,
        target_split=args.split,
        use_corrected=not args.use_original_kp,
    )
