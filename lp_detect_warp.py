#!/usr/bin/env python3
"""
lp_detect_warp.py — 車牌偵測 + 四角投影（無 OCR）
===================================================
輸入一張大圖，輸出：
  1. 每個偵測到的車牌四角座標 (TL, TR, BR, BL)
  2. 透視投影後的車牌小圖存檔

用法：
    python lp_detect_warp.py --model best.pt --image test.jpg
    python lp_detect_warp.py --model best.pt --image test.jpg --output ./out --conf 0.25
    python lp_detect_warp.py --model best.pt --image ./img_dir   # 整個目錄
"""

import os, argparse, json
from pathlib import Path

import numpy as np
import cv2

# ── 參數 ──
IMGSZ    = 640
CONF_DEF = 0.25
PLATE_W  = 400
PLATE_H  = 120


def warp_plate(img_bgr, kpts, target_w=PLATE_W, target_h=PLATE_H):
    """
    四角點透視投影 → 矩形車牌。
    kpts: shape (4, 2)，像素座標
    回傳 (warped_bgr, ordered_pts[TL,TR,BR,BL]) 或 (None, None)
    """
    if np.any(kpts[:, 0] <= 0) or np.any(kpts[:, 1] <= 0):
        return None, None

    center = kpts.mean(axis=0)
    angles = np.arctan2(kpts[:, 1] - center[1], kpts[:, 0] - center[0])
    sorted_kpts = kpts[np.argsort(angles)]

    # 判斷水平 / 垂直車牌
    dists = [np.linalg.norm(sorted_kpts[i] - sorted_kpts[(i + 1) % 4])
             for i in range(4)]
    max_i = int(np.argmax(dists))
    pt1, pt2 = sorted_kpts[max_i], sorted_kpts[(max_i + 1) % 4]
    is_horizontal = abs(pt1[0] - pt2[0]) > abs(pt1[1] - pt2[1])

    if is_horizontal:
        top_idx  = np.argsort(sorted_kpts[:, 1])
        top_pts  = sorted_kpts[top_idx[:2]]
        bot_pts  = sorted_kpts[top_idx[2:]]
        tl = top_pts[np.argmin(top_pts[:, 0])]
        tr = top_pts[np.argmax(top_pts[:, 0])]
        bl = bot_pts[np.argmin(bot_pts[:, 0])]
        br = bot_pts[np.argmax(bot_pts[:, 0])]
    else:
        left_idx  = np.argsort(sorted_kpts[:, 0])
        left_pts  = sorted_kpts[left_idx[:2]]
        right_pts = sorted_kpts[left_idx[2:]]
        tl = left_pts[np.argmin(left_pts[:, 1])]
        bl = left_pts[np.argmax(left_pts[:, 1])]
        tr = right_pts[np.argmin(right_pts[:, 1])]
        br = right_pts[np.argmax(right_pts[:, 1])]

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [target_w - 1, 0],
                    [target_w - 1, target_h - 1], [0, target_h - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, M, (target_w, target_h),
                                 flags=cv2.INTER_LANCZOS4)
    return warped, src


def detect_one(img_path, model, conf_threshold, output_dir):
    """
    對單張圖偵測車牌，存投影小圖，回傳偵測結果 list。
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f'  [WARN] 無法讀取: {img_path}')
        return []

    results = model.predict(source=str(img_path), imgsz=IMGSZ,
                            conf=conf_threshold, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    stem = Path(img_path).stem
    detections = []

    for i in range(len(r.boxes)):
        det_conf = float(r.boxes.conf[i].cpu().numpy())
        if det_conf < conf_threshold:
            continue

        # keypoints
        if r.keypoints is None or len(r.keypoints) <= i:
            continue
        kpts = r.keypoints[i].xy.cpu().numpy()[0]        # (4, 2)
        kp_confs = (r.keypoints[i].conf.cpu().numpy()[0].tolist()
                    if hasattr(r.keypoints[i], 'conf') else [0.0] * 4)

        # 透視投影
        warped, ordered_pts = warp_plate(img_bgr, kpts)
        if warped is None:
            continue

        # 存檔
        plate_name = f'{stem}_{i}.jpg'
        plate_path = output_dir / plate_name
        cv2.imwrite(str(plate_path), warped)

        detections.append({
            'image':     Path(img_path).name,
            'det_conf':  round(det_conf, 4),
            'keypoints': ordered_pts.tolist(),   # TL, TR, BR, BL
            'kp_confs':  [round(c, 4) for c in kp_confs],
            'plate_file': plate_name,
        })

    return detections


def main():
    parser = argparse.ArgumentParser(
        description='車牌偵測 + 四角投影存檔（無 OCR）')
    parser.add_argument('--model',  required=True, help='YOLO-Pose 權重 (.pt)')
    parser.add_argument('--image',  required=True, help='單張圖片或圖片目錄')
    parser.add_argument('--output', default='./lp_crops', help='輸出目錄')
    parser.add_argument('--conf',   type=float, default=CONF_DEF,
                        help=f'偵測信心門檻（預設 {CONF_DEF}）')
    args = parser.parse_args()

    from ultralytics import YOLO
    print(f'載入模型: {args.model}')
    model = YOLO(args.model)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集圖片
    p = Path(args.image)
    if p.is_dir():
        exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        img_files = sorted(f for f in p.iterdir() if f.suffix in exts)
    else:
        img_files = [p]

    print(f'共 {len(img_files)} 張圖片 → {output_dir}\n')

    all_det = []
    for idx, img_path in enumerate(img_files, 1):
        print(f'[{idx}/{len(img_files)}] {img_path.name}', end='')
        dets = detect_one(img_path, model, args.conf, output_dir)
        all_det.extend(dets)
        print(f'  → {len(dets)} 車牌')

    # 寫 JSON 摘要
    json_path = output_dir / 'results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_det, f, ensure_ascii=False, indent=2)

    print(f'\n✅ 完成：{len(all_det)} 個車牌，來自 {len(img_files)} 張圖')
    print(f'   投影小圖 → {output_dir}/')
    print(f'   摘要 JSON → {json_path}')


if __name__ == '__main__':
    main()
