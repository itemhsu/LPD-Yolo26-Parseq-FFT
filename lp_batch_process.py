#!/usr/bin/env python3
"""
車牌批次校正處理器
- 使用 run_one_plate 的核心邏輯（去掉繪圖）
- 在 results.json 每個節點新增：
    h_angle, v_angle, corrected_h_angle, corrected_v_angle,
    corrected_ocr (去掉 -), corrected_plate_img (80x320 base64),
    corrected_conf_first, corrected_conf_last
- 每 N 個節點自動備份，可從備份點接續
- 產生 HTML 分頁檢視器（每頁 100 筆，v角度絕對值降序）
"""

import os, sys, json, math, statistics, copy, base64, time, shutil
from typing import Optional
from pathlib import Path

import numpy as np
import cv2

# ═══════════════════════════════════════════════════════════════
# 可調參數
# ═══════════════════════════════════════════════════════════════
RESUL_JSON   = '/Users/miniaicar/amtk/lp/lp_viewer_tool/lp_viewer_output/results.json'
OUTPUT_DIR   = '/Users/miniaicar/amtk/lp/lp_viewer_tool/lp_viewer_output/'
BACKUP_JSON  = os.path.join(OUTPUT_DIR, 'results_backup.json')
HTML_OUTPUT  = os.path.join(OUTPUT_DIR, 'plate_viewer.html')

RECT_W = 640
RECT_H = 160
OUT_W  = 320    # 輸出縮圖寬
OUT_H  = 80     # 輸出縮圖高

STEP_FRAC     = 0.02
MAX_ITER      = 30
ANGLE_STOP    = 0.001
OCR_CONF_STOP = 0.99
LEFT_EXPAND_FRAC = 0.01
LEFT_EXPAND_MAX  = 10

BATCH_SIZE      = 50     # 每次處理多少個節點
CHECKPOINT_EVERY = 10    # 每處理幾個就備份一次

PATH_MAP = {'/home/itemhsu': '/Users/miniaicar'}

FFT_IS_PLOT  = False
FFT_IS_BIAS  = False
FFT_CORE_CUT = 0.01

# ═══════════════════════════════════════════════════════════════
# PARSeq OCR (延遲初始化)
# ═══════════════════════════════════════════════════════════════
_parseq_model = None
_parseq_device = None
_parseq_transform = None

def _init_parseq():
    global _parseq_model, _parseq_device, _parseq_transform
    if _parseq_model is not None:
        return
    # 自動安裝缺少的依賴
    import importlib, subprocess
    for pkg, pip_name in [('pytorch_lightning', 'pytorch_lightning'),
                          ('timm', 'timm'),
                          ('nltk', 'nltk')]:
        try:
            importlib.import_module(pkg)
        except ImportError:
            print(f'Installing missing dependency: {pip_name}')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
    import torch
    from torchvision import transforms
    _parseq_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _parseq_model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True)
    _parseq_model = _parseq_model.eval().to(_parseq_device)
    _parseq_transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    print(f'PARSeq loaded on {_parseq_device}')

def process_roi(plate_bgr: np.ndarray) -> list:
    import torch
    from PIL import Image
    _init_parseq()
    pil_img = Image.fromarray(cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB))
    inp = _parseq_transform(pil_img).unsqueeze(0).to(_parseq_device)
    with torch.no_grad():
        logits = _parseq_model(inp)
        probs = logits.softmax(-1)
    preds = _parseq_model.tokenizer.decode(probs)
    if isinstance(preds, (tuple, list)) and len(preds) == 2:
        texts, confs = preds
    else:
        texts, confs = preds, None
    text = texts[0] if isinstance(texts, (list, tuple)) else texts
    char_probs = probs[0]
    per_char_conf = char_probs.max(-1).values
    chars = []
    for idx, ch in enumerate(text):
        if confs is not None:
            try:
                conf = float(confs[0][idx])
            except Exception:
                conf = float(per_char_conf[idx]) if idx < len(per_char_conf) else 0.0
        else:
            conf = float(per_char_conf[idx]) if idx < len(per_char_conf) else 0.0
        chars.append({'char': ch, 'conf': conf})
    return chars

# ═══════════════════════════════════════════════════════════════
# 路徑 / 幾何 / FFT 工具
# ═══════════════════════════════════════════════════════════════
def _map_path(p: str) -> str:
    for src, dst in PATH_MAP.items():
        if p.startswith(src):
            return p.replace(src, dst, 1)
    return p

def resolve_image_path(rec: dict) -> Optional[str]:
    for key in ('original_path', 'original_image'):
        p = rec.get(key, '')
        if not p:
            continue
        for cand in [p, _map_path(p)]:
            if os.path.exists(cand):
                return cand
    return None

def order_quad_tl_tr_br_bl(quad: np.ndarray) -> np.ndarray:
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    s = q[:, 0] + q[:, 1]
    d = q[:, 0] - q[:, 1]
    return np.stack([
        q[np.argmin(s)], q[np.argmax(d)],
        q[np.argmax(s)], q[np.argmin(d)],
    ]).astype(np.float32)

def quad_from_keypoints(kps) -> np.ndarray:
    q = np.array(kps, dtype=np.float32)
    xs = np.argsort(q[:, 0])
    left = q[xs[:2]]; left = left[np.argsort(left[:, 1])]
    right = q[xs[2:]]; right = right[np.argsort(right[:, 1])]
    return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)

def rectify_plate(img_bgr, quad, rect_w=RECT_W, rect_h=RECT_H):
    q = order_quad_tl_tr_br_bl(quad)
    dst = np.array([[0,0],[rect_w-1,0],[rect_w-1,rect_h-1],[0,rect_h-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(q, dst)
    plate = cv2.warpPerspective(img_bgr, H, (rect_w, rect_h),
                                flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_REPLICATE)
    return plate, H

def _unit_vec(a, b):
    v = b - a
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros_like(v)

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def imgWrapA(orgImg, a):
    col, row = orgImg.shape[1], orgImg.shape[0]
    pts1 = np.float32([[col/2, row/2], [col/2, row/4], [col/4, row/2 - col/4*a]])
    pts2 = np.float32([[col/2, row/2], [col/2, row/4], [col/4, row/2]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(orgImg, M, (col, row))

def estCorrect(orgImg0, cutoffF=0.8, margin=0.1):
    orgImg = cv2.fastNlMeansDenoisingColored(orgImg0, None, 9, 9, 7, 21)
    w = orgImg.shape[1]
    crop_img = orgImg[:, int(margin*w):int(w - margin*w)]
    img = rgb2gray(np.array(crop_img))
    img = abs(img[0:-1, :] - img[1:, :])
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)
    size = min(img.shape)
    polar_img = cv2.warpPolar(
        magnitude_spectrum, (int(size/2), 200),
        (img.shape[1]/2, img.shape[0]/2),
        size * 0.9 * 0.5, cv2.WARP_POLAR_LINEAR)
    polar_img_lowF = polar_img[:, int(FFT_CORE_CUT * polar_img.shape[1]):int(cutoffF * polar_img.shape[1])]
    polar_sum_200 = np.sum(polar_img_lowF, axis=1)
    polar_sum = polar_sum_200[0:100] + polar_sum_200[100:200]
    if FFT_IS_BIAS:
        gain_stdev = statistics.stdev(polar_sum[25:75]) / 10000
        polar_sum[45:56] = polar_sum[45:56] * gain_stdev + polar_sum[45:56] * (1 - gain_stdev)
    maxIndex = np.argmax(polar_sum[25:75]) + 25
    offsetDegree = (maxIndex - 50) / 100 * 3.14
    aEst = np.sin(offsetDegree)
    correctImg = imgWrapA(np.array(orgImg0), aEst)
    return correctImg, float(offsetDegree)

def preprocess_for_fft(orgImg0):
    denoised = cv2.fastNlMeansDenoisingColored(orgImg0, None, 9, 9, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

def estCorrect2D(orgImg, cutoffF=0.8, margin=0.1):
    h, w = orgImg.shape[:2]
    side = int(round(math.sqrt(w * h)))
    orgImg = preprocess_for_fft(orgImg)
    squareImg = cv2.resize(orgImg, (side, side), interpolation=cv2.INTER_LINEAR)
    hCorrectedImg, hOffsetDegree = estCorrect(squareImg, cutoffF, margin)
    hCorrectedImg90 = np.rot90(hCorrectedImg)
    vCorrectedImg, vOffsetDegree = estCorrect(hCorrectedImg90, cutoffF, margin)
    vCorrectedImg270 = np.rot90(vCorrectedImg, k=3)
    return hCorrectedImg, vCorrectedImg270, float(hOffsetDegree), float(vOffsetDegree)

def analyse_plate_angle(plate_bgr):
    h_img, hv_img, h_angle, v_angle = estCorrect2D(plate_bgr, 0.8, 0.1)
    return dict(hOffsetDegree=float(h_angle), vOffsetDegree=float(v_angle),
                hCorrectedImg=h_img, correctedImg=hv_img)

def refine_quad_one_step(q, info, step_frac=STEP_FRAC):
    q_new = q.astype(np.float32).copy()
    TL, TR, BR, BL = q_new[0], q_new[1], q_new[2], q_new[3]
    u_left  = _unit_vec(TL, BL); u_right = _unit_vec(TR, BR)
    u_top   = _unit_vec(TL, TR); u_bot   = _unit_vec(BL, BR)
    len_left  = float(np.linalg.norm(BL - TL)) + 1e-6
    len_right = float(np.linalg.norm(BR - TR)) + 1e-6
    len_top   = float(np.linalg.norm(TR - TL)) + 1e-6
    len_bot   = float(np.linalg.norm(BR - BL)) + 1e-6
    ml = step_frac * len_left;  mr = step_frac * len_right
    mt = step_frac * len_top;   mb = step_frac * len_bot
    h = float(info['hOffsetDegree']); v = float(info['vOffsetDegree'])
    moves = []
    if h < -ANGLE_STOP:
        q_new[1] -= mr * u_right; q_new[3] += ml * u_left
        moves.append(f'h={h:+.6f}')
    elif h > ANGLE_STOP:
        q_new[0] -= ml * u_left; q_new[2] += mr * u_right
        moves.append(f'h={h:+.6f}')
    if v > ANGLE_STOP:
        q_new[1] += mt * u_top; q_new[3] -= mb * u_bot
        moves.append(f'v={v:+.6f}')
    elif v < -ANGLE_STOP:
        q_new[0] -= mt * u_top; q_new[2] += mb * u_bot
        moves.append(f'v={v:+.6f}')
    return q_new, ' | '.join(moves) if moves else 'converged'

def expand_left_edge(q, frac=LEFT_EXPAND_FRAC):
    q_new = q.astype(np.float32).copy()
    u_top = _unit_vec(q_new[0], q_new[1]); u_bot = _unit_vec(q_new[3], q_new[2])
    lt = float(np.linalg.norm(q_new[1] - q_new[0])) + 1e-6
    lb = float(np.linalg.norm(q_new[2] - q_new[3])) + 1e-6
    q_new[0] -= frac * lt * u_top; q_new[3] -= frac * lb * u_bot
    return q_new

def expand_right_edge(q, frac=LEFT_EXPAND_FRAC):
    q_new = q.astype(np.float32).copy()
    u_top = _unit_vec(q_new[0], q_new[1]); u_bot = _unit_vec(q_new[3], q_new[2])
    lt = float(np.linalg.norm(q_new[1] - q_new[0])) + 1e-6
    lb = float(np.linalg.norm(q_new[2] - q_new[3])) + 1e-6
    q_new[1] += frac * lt * u_top; q_new[2] += frac * lb * u_bot
    return q_new

def analyse_parseq_conf(plate_bgr):
    chars = process_roi(plate_bgr)
    if not chars:
        return dict(chars=[], ocr_str='', conf_first=0.0, conf_last=0.0,
                    pass_first=False, pass_last=False, pass_both=False)
    ocr_str = ''.join(r.get('char', '') for r in chars)
    cf = float(chars[0].get('conf', 0.0))
    cl = float(chars[-1].get('conf', 0.0))
    return dict(chars=chars, ocr_str=ocr_str, conf_first=cf, conf_last=cl,
                pass_first=cf >= OCR_CONF_STOP, pass_last=cl >= OCR_CONF_STOP,
                pass_both=(cf >= OCR_CONF_STOP and cl >= OCR_CONF_STOP))

# ═══════════════════════════════════════════════════════════════
# 圖片轉 base64
# ═══════════════════════════════════════════════════════════════
def img_to_base64(img_bgr, w=OUT_W, h=OUT_H):
    resized = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode('.png', resized)
    return base64.b64encode(buf).decode('utf-8')

def draw_quad_on_img(img, quad, color=(0,255,0), thickness=2):
    vis = img.copy()
    pts = quad.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], True, color, thickness)
    return vis

def crop_around_quad(img, quad, pad=60):
    H_img, W_img = img.shape[:2]
    x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
    y0 = max(0, y - pad); y1 = min(H_img, y + h + pad)
    x0 = max(0, x - pad); x1 = min(W_img, x + w + pad)
    return img[y0:y1, x0:x1] if y1 > y0 and x1 > x0 else img

# ═══════════════════════════════════════════════════════════════
# 核心處理：精簡版 run_one_plate（無繪圖）
# ═══════════════════════════════════════════════════════════════
def process_one_plate(rec, resul_by_id):
    """處理一個車牌，回傳要寫入 results.json 的新增欄位 dict，或 None"""
    plate_id = rec.get('plate_id', '')
    if 'keypoints' not in rec:
        return None

    img_path = resolve_image_path(rec)
    if img_path is None or not os.path.exists(img_path):
        print(f'  [{plate_id}] image not found, skip')
        return None

    img = cv2.imread(img_path)
    if img is None:
        print(f'  [{plate_id}] cv2.imread failed, skip')
        return None

    quad = quad_from_keypoints(rec['keypoints'])
    quad_orig = quad.copy()

    # --- baseline ---
    plate_orig, _ = rectify_plate(img, quad)
    info = analyse_plate_angle(plate_orig)
    baseline_h = info['hOffsetDegree']
    baseline_v = info['vOffsetDegree']

    # --- iterative quad refinement ---
    for it in range(1, MAX_ITER + 1):
        if abs(info['hOffsetDegree']) <= ANGLE_STOP and abs(info['vOffsetDegree']) <= ANGLE_STOP:
            break
        quad_new, _ = refine_quad_one_step(quad, info)
        plate_new, _ = rectify_plate(img, quad_new)
        info = analyse_plate_angle(plate_new)
        quad = quad_new

    plate_corrected, _ = rectify_plate(img, quad)

    # --- OCR + edge expand ---
    ocr = analyse_parseq_conf(plate_corrected)
    best_cf = ocr['conf_first']; best_cl = ocr['conf_last']

    for ex in range(1, LEFT_EXPAND_MAX + 1):
        if ocr['pass_both']:
            break
        need_left = not ocr['pass_first']
        need_right = not ocr['pass_last']
        q_new = quad.copy()
        if need_left:
            q_new = expand_left_edge(q_new)
        if need_right:
            q_new = expand_right_edge(q_new)
        plate_new, _ = rectify_plate(img, q_new)
        ocr_new = analyse_parseq_conf(plate_new)

        improved_l = need_left  and (ocr_new['conf_first'] > best_cf)
        improved_r = need_right and (ocr_new['conf_last']  > best_cl)
        if need_left:  best_cf = max(best_cf, ocr_new['conf_first'])
        if need_right: best_cl = max(best_cl, ocr_new['conf_last'])
        if not improved_l and not improved_r:
            break
        quad = q_new
        plate_corrected, _ = rectify_plate(img, quad)
        ocr = ocr_new

    # --- 組裝結果 ---
    corrected_ocr = ocr['ocr_str'].replace('-', '')
    corrected_h = info['hOffsetDegree']
    corrected_v = info['vOffsetDegree']

    # 圖片：原始投影 & 校正後投影 (80x320 base64)
    plate_orig_img = plate_orig  # 640x160
    plate_corr_img = plate_corrected

    orig_b64 = img_to_base64(plate_orig_img)
    corr_b64 = img_to_base64(plate_corr_img)

    # 原圖上畫 quad 的裁切區
    vis_orig_quad = draw_quad_on_img(img, quad_orig, (0, 0, 255), 3)
    vis_corr_quad = draw_quad_on_img(vis_orig_quad, quad, (0, 255, 0), 3)
    crop_vis = crop_around_quad(vis_corr_quad, quad_orig, pad=80)
    crop_b64 = img_to_base64(crop_vis, w=480, h=int(480 * crop_vis.shape[0] / max(crop_vis.shape[1], 1)))

    print(f'  [{plate_id}] h={baseline_h:+.4f}->{corrected_h:+.4f}  '
          f'v={baseline_v:+.4f}->{corrected_v:+.4f}  '
          f'OCR={corrected_ocr}  cf={ocr["conf_first"]:.3f}  cl={ocr["conf_last"]:.3f}')

    return {
        'h_angle': round(baseline_h, 6),
        'v_angle': round(baseline_v, 6),
        'corrected_h_angle': round(corrected_h, 6),
        'corrected_v_angle': round(corrected_v, 6),
        'corrected_ocr': corrected_ocr,
        'corrected_plate_img_b64': corr_b64,
        'original_plate_img_b64': orig_b64,
        'corrected_conf_first': round(ocr['conf_first'], 6),
        'corrected_conf_last': round(ocr['conf_last'], 6),
        'quad_crop_img_b64': crop_b64,
        'corrected_quad': quad.tolist(),
    }

# ═══════════════════════════════════════════════════════════════
# 備份 / 讀取
# ═══════════════════════════════════════════════════════════════
def save_results(data, path=RESUL_JSON):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    shutil.move(tmp, path)
    print(f'  [saved] {path}  ({len(data)} records)')

def save_backup(data):
    save_results(data, BACKUP_JSON)

def load_results():
    # 優先從備份讀取
    target = BACKUP_JSON if os.path.exists(BACKUP_JSON) else RESUL_JSON
    print(f'Loading from: {target}')
    with open(target, 'r', encoding='utf-8') as f:
        return json.load(f)

# ═══════════════════════════════════════════════════════════════
# 主批次處理
# ═══════════════════════════════════════════════════════════════
def batch_process(max_plates=BATCH_SIZE, checkpoint_every=CHECKPOINT_EVERY):
    data = load_results()
    resul_by_id = {r['plate_id']: r for r in data if 'plate_id' in r}

    # 找出尚未處理的
    pending = [r for r in data if 'plate_id' in r and 'corrected_ocr' not in r and 'keypoints' in r]
    to_process = pending[:max_plates]
    print(f'Total records: {len(data)}, pending: {len(pending)}, this batch: {len(to_process)}')

    processed = 0
    for i, rec in enumerate(to_process):
        pid = rec['plate_id']
        print(f'\n[{i+1}/{len(to_process)}] Processing {pid} ...')
        try:
            result = process_one_plate(rec, resul_by_id)
            if result:
                # 直接更新 data 中的 record（同一個 dict 物件）
                rec.update(result)
                processed += 1
        except Exception as e:
            print(f'  [ERROR] {pid}: {e}')
            import traceback; traceback.print_exc()
            continue

        # checkpoint
        if processed > 0 and processed % checkpoint_every == 0:
            print(f'\n--- Checkpoint at {processed} processed ---')
            save_backup(data)

    # 最終儲存
    if processed > 0:
        save_results(data, RESUL_JSON)
        save_backup(data)
    print(f'\nDone. Processed {processed}/{len(to_process)} plates.')
    return data

# ═══════════════════════════════════════════════════════════════
# HTML 產生器 — 10x10 網格版
# ═══════════════════════════════════════════════════════════════
def generate_html(data=None, output_path=HTML_OUTPUT, per_page=100):
    if data is None:
        data = load_results()

    items = [r for r in data if 'corrected_ocr' in r]
    items.sort(key=lambda r: abs(r.get('v_angle', 0)), reverse=True)

    total = len(items)
    total_pages = max(1, (total + per_page - 1) // per_page)
    print(f'Generating HTML: {total} items, {total_pages} pages')

    pages_data = []
    for p in range(total_pages):
        start = p * per_page
        end = min(start + per_page, total)
        page_items = []
        for r in items[start:end]:
            page_items.append({
                'plate_id': r.get('plate_id', ''),
                'h_angle': r.get('h_angle', 0),
                'v_angle': r.get('v_angle', 0),
                'corrected_h_angle': r.get('corrected_h_angle', 0),
                'corrected_v_angle': r.get('corrected_v_angle', 0),
                'corrected_ocr': r.get('corrected_ocr', ''),
                'corrected_conf_first': r.get('corrected_conf_first', 0),
                'corrected_conf_last': r.get('corrected_conf_last', 0),
                'original_plate_img_b64': r.get('original_plate_img_b64', ''),
                'corrected_plate_img_b64': r.get('corrected_plate_img_b64', ''),
                'quad_crop_img_b64': r.get('quad_crop_img_b64', ''),
                'ocr': r.get('ocr', ''),
            })
        pages_data.append(page_items)

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>車牌校正結果檢視器</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #111; color: #eee; padding: 10px; }}
.header {{ text-align: center; padding: 8px 0; }}
.header h1 {{ font-size: 18px; color: #e94560; margin-bottom: 4px; }}
.header .info {{ color: #888; font-size: 12px; }}

/* Pagination */
.pagination {{ display: flex; justify-content: center; gap: 4px; margin: 8px 0; flex-wrap: wrap; }}
.pagination button {{
    background: #1a1a2e; color: #ccc; border: 1px solid #333; padding: 4px 10px;
    cursor: pointer; border-radius: 3px; font-size: 12px;
}}
.pagination button.active {{ background: #e94560; color: #fff; border-color: #e94560; font-weight: bold; }}
.pagination button:hover:not(:disabled) {{ background: #0f3460; }}
.pagination button:disabled {{ opacity: 0.4; cursor: default; }}

/* 10x10 Grid */
.grid {{
    display: grid;
    grid-template-columns: repeat(10, 1fr);
    gap: 4px;
    margin: 8px 0;
}}

/* Each cell */
.cell {{
    background: #1a1a2e;
    border: 1px solid #222;
    border-radius: 4px;
    cursor: pointer;
    overflow: hidden;
    transition: border-color 0.15s, transform 0.15s;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 3px 2px;
}}
.cell:hover {{ border-color: #e94560; transform: scale(1.03); z-index: 2; }}
.cell .name {{
    font-size: 10px;
    font-weight: bold;
    color: #e94560;
    text-align: center;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    width: 100%;
    padding: 1px 2px;
    line-height: 1.3;
}}
.cell img {{
    width: 100%;
    height: auto;
    display: block;
    image-rendering: auto;
}}
.cell .lbl {{
    font-size: 8px;
    color: #666;
    text-align: center;
    line-height: 1.2;
}}

/* Modal overlay */
.modal-overlay {{
    display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.88); z-index: 1000; justify-content: center; align-items: center;
}}
.modal-overlay.active {{ display: flex; }}
.modal {{
    background: #16213e; border-radius: 10px; padding: 20px 24px;
    max-width: 850px; width: 95%; max-height: 92vh; overflow-y: auto; position: relative;
}}
.modal .close {{
    position: absolute; top: 8px; right: 14px; font-size: 26px;
    color: #e94560; cursor: pointer; line-height: 1;
}}
.modal .close:hover {{ color: #ff6b81; }}
.modal h2 {{ color: #e94560; margin-bottom: 14px; font-size: 20px; }}
.modal table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; }}
.modal th {{ color: #e94560; text-align: left; padding: 5px 8px; font-size: 13px; border-bottom: 1px solid #0f3460; }}
.modal td {{ padding: 5px 8px; font-size: 13px; border-bottom: 1px solid #0f3460; color: #ccc; }}
.modal .img-section {{ margin-bottom: 16px; }}
.modal .img-section h3 {{ color: #aaa; font-size: 13px; margin-bottom: 6px; }}
.modal .img-section img {{ max-width: 100%; border-radius: 5px; border: 1px solid #333; }}
.modal .plate-compare {{
    display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap;
}}
.modal .plate-compare .col {{ flex: 1; min-width: 200px; }}
.modal .plate-compare .col h3 {{ color: #aaa; font-size: 13px; margin-bottom: 6px; }}
.modal .plate-compare .col img {{ width: 100%; border-radius: 5px; border: 1px solid #333; }}
</style>
</head>
<body>

<div class="header">
    <h1>車牌校正結果檢視器</h1>
    <div class="info">共 {total} 筆 ｜ |v_angle| 降序 ｜ 每頁 {per_page} 筆 ｜ 點擊查看詳情</div>
</div>
<div class="pagination" id="pagination"></div>
<div class="grid" id="grid"></div>
<div class="pagination" id="pagination2"></div>

<div class="modal-overlay" id="modalOverlay" onclick="if(event.target===this)closeModal()">
  <div class="modal" id="modal"></div>
</div>

<script>
const PAGES = {json.dumps(pages_data, ensure_ascii=False)};
const TOTAL_PAGES = {total_pages};
const PER_PAGE = {per_page};
let currentPage = 0;

function buildPaginationHTML() {{
    let h = '';
    h += `<button onclick="goPage(${{currentPage-1}})" ${{currentPage===0?'disabled':''}}>◀</button>`;
    const show = new Set();
    for (let i=0; i<Math.min(2,TOTAL_PAGES); i++) show.add(i);
    for (let i=Math.max(0,TOTAL_PAGES-2); i<TOTAL_PAGES; i++) show.add(i);
    for (let i=Math.max(0,currentPage-2); i<=Math.min(TOTAL_PAGES-1,currentPage+2); i++) show.add(i);
    const sorted=[...show].sort((a,b)=>a-b);
    let prev=-1;
    for (const p of sorted) {{
        if (prev>=0 && p-prev>1) h+='<button disabled>…</button>';
        h+=`<button class="${{p===currentPage?'active':''}}" onclick="goPage(${{p}})">${{p+1}}</button>`;
        prev=p;
    }}
    h += `<button onclick="goPage(${{currentPage+1}})" ${{currentPage===TOTAL_PAGES-1?'disabled':''}}>▶</button>`;
    return h;
}}

function renderPage() {{
    const pgHTML = buildPaginationHTML();
    document.getElementById('pagination').innerHTML = pgHTML;
    document.getElementById('pagination2').innerHTML = pgHTML;

    const items = PAGES[currentPage] || [];
    const el = document.getElementById('grid');
    el.innerHTML = items.map((r, idx) => `
        <div class="cell" onclick="showModal(${{currentPage}},${{idx}})">
            <div class="name">${{r.corrected_ocr || r.ocr || '—'}}</div>
            <div class="lbl">原始</div>
            <img src="data:image/png;base64,${{r.original_plate_img_b64}}" alt="orig">
            <div class="lbl">校正</div>
            <img src="data:image/png;base64,${{r.corrected_plate_img_b64}}" alt="corr">
        </div>
    `).join('');
}}

function goPage(p) {{
    if (p<0||p>=TOTAL_PAGES) return;
    currentPage=p;
    renderPage();
    window.scrollTo(0,0);
}}

function showModal(page, idx) {{
    const r = PAGES[page][idx];
    const globalIdx = page * PER_PAGE + idx + 1;
    const modal = document.getElementById('modal');
    modal.innerHTML = `
        <span class="close" onclick="closeModal()">&times;</span>
        <h2>#${{globalIdx}} ${{r.corrected_ocr || r.ocr || '—'}}</h2>
        <table>
            <tr><th>項目</th><th>校正前</th><th>校正後</th></tr>
            <tr><td>h 角度</td><td>${{r.h_angle.toFixed(6)}}</td><td>${{r.corrected_h_angle.toFixed(6)}}</td></tr>
            <tr><td>v 角度</td><td>${{r.v_angle.toFixed(6)}}</td><td>${{r.corrected_v_angle.toFixed(6)}}</td></tr>
            <tr><td>|v| 角度</td><td>${{Math.abs(r.v_angle).toFixed(6)}}</td><td>${{Math.abs(r.corrected_v_angle).toFixed(6)}}</td></tr>
            <tr><td>首字信心</td><td>—</td><td>${{r.corrected_conf_first.toFixed(6)}}</td></tr>
            <tr><td>尾字信心</td><td>—</td><td>${{r.corrected_conf_last.toFixed(6)}}</td></tr>
            <tr><td>Plate ID</td><td colspan="2" style="font-size:11px;color:#888;">${{r.plate_id}}</td></tr>
        </table>
        <div class="plate-compare">
            <div class="col">
                <h3>校正前投影</h3>
                <img src="data:image/png;base64,${{r.original_plate_img_b64}}">
            </div>
            <div class="col">
                <h3>校正後投影</h3>
                <img src="data:image/png;base64,${{r.corrected_plate_img_b64}}">
            </div>
        </div>
        <div class="img-section">
            <h3>原圖框定位置（紅＝原始  綠＝校正後）</h3>
            <img src="data:image/png;base64,${{r.quad_crop_img_b64}}">
        </div>
    `;
    document.getElementById('modalOverlay').classList.add('active');
}}

function closeModal() {{
    document.getElementById('modalOverlay').classList.remove('active');
}}

document.addEventListener('keydown', e => {{
    if (e.key==='Escape') closeModal();
    if (e.key==='ArrowLeft') goPage(currentPage-1);
    if (e.key==='ArrowRight') goPage(currentPage+1);
}});

goPage(0);
</script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'HTML saved: {output_path}')

# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='車牌批次校正處理器')
    parser.add_argument('--max', type=int, default=BATCH_SIZE,
                        help=f'本次最多處理幾個節點 (default: {BATCH_SIZE})')
    parser.add_argument('--checkpoint', type=int, default=CHECKPOINT_EVERY,
                        help=f'每幾個備份一次 (default: {CHECKPOINT_EVERY})')
    parser.add_argument('--html-only', action='store_true',
                        help='只產生 HTML 不處理新的')
    parser.add_argument('--json', type=str, default=RESUL_JSON,
                        help='results.json 路徑')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='輸出目錄')
    args = parser.parse_args()

    RESUL_JSON = args.json
    OUTPUT_DIR = args.output_dir
    BACKUP_JSON = os.path.join(OUTPUT_DIR, 'results_backup.json')
    HTML_OUTPUT = os.path.join(OUTPUT_DIR, 'plate_viewer.html')

    if args.html_only:
        generate_html()
    else:
        data = batch_process(max_plates=args.max, checkpoint_every=args.checkpoint)
        generate_html(data)
