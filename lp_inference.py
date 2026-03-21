#!/usr/bin/env python3
"""
lp_inference.py — 車牌偵測、OCR 與 HTML 檢視器
=================================================
三個主要函數：
  1. infer_one(img_path, model, conf_threshold) → 對單張圖執行 YOLO-Pose + PARSeq OCR
  2. infer_dir(input_dir, output_dir, model_path, conf_threshold) → 批次處理目錄
  3. generate_html(json_path, output_html) → 產生分頁 HTML 檢視器

用法：
    python lp_inference.py --model best.pt --input ./img --output ./results/
    python lp_inference.py --model best.pt --input ./img --output ./results/ --html-only
"""

import os, sys, json, base64, argparse, time
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

# ═══════════════════════════════════════════════════════════════
# 可調參數
# ═══════════════════════════════════════════════════════════════
IMGSZ         = 640
CONF_THRESHOLD = 0.01
PLATE_W       = 400
PLATE_H       = 120

# ═══════════════════════════════════════════════════════════════
# PARSeq OCR（延遲初始化）
# ═══════════════════════════════════════════════════════════════
_parseq_model     = None
_parseq_device    = None
_parseq_transform = None


def _init_parseq():
    global _parseq_model, _parseq_device, _parseq_transform
    if _parseq_model is not None:
        return
    import torch
    from torchvision import transforms
    _parseq_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading PARSeq OCR on {_parseq_device}...')
    _parseq_model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True)
    _parseq_model = _parseq_model.eval().to(_parseq_device)
    _parseq_transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    print(f'✅ PARSeq loaded on {_parseq_device}')


def _ocr_plate(plate_bgr: np.ndarray) -> dict:
    """
    對車牌 BGR 影像執行 PARSeq OCR。
    回傳 {'text', 'chars', 'conf_first', 'conf_last'}
    """
    import torch
    from PIL import Image
    _init_parseq()

    pil_img = Image.fromarray(cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB))
    inp = _parseq_transform(pil_img).unsqueeze(0).to(_parseq_device)

    with torch.no_grad():
        logits = _parseq_model(inp)
        probs  = logits.softmax(-1)

    preds = _parseq_model.tokenizer.decode(probs)
    if isinstance(preds, (tuple, list)) and len(preds) == 2:
        texts, confs = preds
    else:
        texts, confs = preds, None

    text = texts[0] if isinstance(texts, (list, tuple)) else texts
    per_char_conf = probs[0].max(-1).values

    chars = []
    for idx, ch in enumerate(text):
        if confs is not None:
            try:
                conf = float(confs[0][idx])
            except Exception:
                conf = float(per_char_conf[idx]) if idx < len(per_char_conf) else 0.0
        else:
            conf = float(per_char_conf[idx]) if idx < len(per_char_conf) else 0.0
        chars.append({'char': ch, 'conf': round(conf, 6)})

    conf_first = chars[0]['conf']  if chars else 0.0
    conf_last  = chars[-1]['conf'] if chars else 0.0

    return {
        'text':       text,
        'chars':      chars,
        'conf_first': round(conf_first, 6),
        'conf_last':  round(conf_last,  6),
    }


# ═══════════════════════════════════════════════════════════════
# 透視投影工具
# ═══════════════════════════════════════════════════════════════

def _warp_plate(img_bgr: np.ndarray, kpts: np.ndarray,
                target_w: int = PLATE_W, target_h: int = PLATE_H):
    """
    將四角點透視投影成矩形車牌。
    kpts: shape (4, 2) — TL TR BR BL
    回傳 (warped_bgr, src_pts) 或 (None, None)
    """
    if not (np.all(kpts[:, 0] > 0) and np.all(kpts[:, 1] > 0)):
        return None, None

    center = kpts.mean(axis=0)
    angles = np.arctan2(kpts[:, 1] - center[1], kpts[:, 0] - center[0])
    sorted_kpts = kpts[np.argsort(angles)]

    # 判斷方向
    dists = [np.linalg.norm(sorted_kpts[i] - sorted_kpts[(i+1) % 4]) for i in range(4)]
    max_i = int(np.argmax(dists))
    pt1, pt2 = sorted_kpts[max_i], sorted_kpts[(max_i + 1) % 4]
    is_horizontal = abs(pt1[0] - pt2[0]) > abs(pt1[1] - pt2[1])

    if is_horizontal:
        top_idx    = np.argsort(sorted_kpts[:, 1])
        top_pts    = sorted_kpts[top_idx[:2]]
        bot_pts    = sorted_kpts[top_idx[2:]]
        top_left   = top_pts[np.argmin(top_pts[:, 0])]
        top_right  = top_pts[np.argmax(top_pts[:, 0])]
        bot_left   = bot_pts[np.argmin(bot_pts[:, 0])]
        bot_right  = bot_pts[np.argmax(bot_pts[:, 0])]
    else:
        left_idx   = np.argsort(sorted_kpts[:, 0])
        left_pts   = sorted_kpts[left_idx[:2]]
        right_pts  = sorted_kpts[left_idx[2:]]
        top_left   = left_pts[np.argmin(left_pts[:, 1])]
        bot_left   = left_pts[np.argmax(left_pts[:, 1])]
        top_right  = right_pts[np.argmin(right_pts[:, 1])]
        bot_right  = right_pts[np.argmax(right_pts[:, 1])]

    src_pts = np.array([top_left, top_right, bot_right, bot_left], dtype=np.float32)
    dst_pts = np.array([[0, 0], [target_w-1, 0],
                        [target_w-1, target_h-1], [0, target_h-1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img_bgr, M, (target_w, target_h))
    return warped, src_pts


def _img_to_b64(img_bgr: np.ndarray, fmt: str = '.jpg', quality: int = 85) -> str:
    """將 BGR 影像編碼為 base64 字串（data URI 不含前綴）"""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == '.jpg' else []
    ok, buf = cv2.imencode(fmt, img_bgr, params)
    if not ok:
        return ''
    return base64.b64encode(buf.tobytes()).decode('ascii')


# ═══════════════════════════════════════════════════════════════
# 函數 1：單張圖推論
# ═══════════════════════════════════════════════════════════════

def infer_one(img_path: str, model, conf_threshold: float = CONF_THRESHOLD) -> list:
    """
    對單張圖執行 YOLO-Pose 偵測 + PARSeq OCR。

    Args:
        img_path:       圖片路徑
        model:          已載入的 YOLO 模型物件
        conf_threshold: 偵測信心門檻

    Returns:
        list[dict]，每個偵測到的車牌回傳一筆：
        {
          'image':        原圖檔名
          'det_conf':     偵測信心
          'keypoints':    [[x,y], [x,y], [x,y], [x,y]]  # TL TR BR BL
          'kp_confs':     [float x4]  # 各角點信心
          'ocr_text':     str
          'conf_first':   float
          'conf_last':    float
          'chars':        [{'char': str, 'conf': float}, ...]
          'plate_b64':    str  # 投影後車牌 base64 (jpg)
          'anno_b64':     str  # 原圖標注（含框線與角點）base64 (jpg)
        }
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f'[WARN] Cannot read: {img_path}')
        return []

    results = model.predict(source=str(img_path), imgsz=IMGSZ,
                            conf=conf_threshold, verbose=False)
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return []

    # 繪製標注圖（用於 Modal 顯示）
    anno_bgr = img_bgr.copy()
    corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    detections = []
    for i in range(len(r.boxes)):
        det_conf = float(r.boxes.conf[i].cpu().numpy())
        if det_conf < conf_threshold:
            continue

        # bounding box
        xyxy = r.boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(anno_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(anno_bgr, f'LP {det_conf:.2f}',
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # keypoints
        if r.keypoints is None or len(r.keypoints) <= i:
            continue
        kpts     = r.keypoints[i].xy.cpu().numpy()[0]   # (4, 2)
        kp_confs = (r.keypoints[i].conf.cpu().numpy()[0].tolist()
                    if hasattr(r.keypoints[i], 'conf') else [0.0] * 4)

        # 繪製角點與連線
        labels = ['TL', 'TR', 'BR', 'BL']
        for j, (x, y) in enumerate(kpts):
            if x > 0 and y > 0:
                cv2.circle(anno_bgr, (int(x), int(y)), 6, corner_colors[j], -1)
                cv2.circle(anno_bgr, (int(x), int(y)), 8, (255, 255, 255), 1)
                cv2.putText(anno_bgr, labels[j],
                            (int(x)+10, int(y)+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for s, e in [(0,1),(1,2),(2,3),(3,0)]:
            xs, ys = kpts[s]; xe, ye = kpts[e]
            if xs > 0 and ys > 0 and xe > 0 and ye > 0:
                cv2.line(anno_bgr,
                         (int(xs), int(ys)), (int(xe), int(ye)),
                         (255, 0, 255), 2)

        # 透視投影
        warped, _ = _warp_plate(img_bgr, kpts)
        if warped is None:
            continue

        # OCR
        ocr = _ocr_plate(warped)

        detections.append({
            'image':      Path(img_path).name,
            'det_conf':   round(det_conf, 6),
            'keypoints':  kpts.tolist(),
            'kp_confs':   [round(float(c), 6) for c in kp_confs],
            'ocr_text':   ocr['text'],
            'conf_first': ocr['conf_first'],
            'conf_last':  ocr['conf_last'],
            'chars':      ocr['chars'],
            'plate_b64':  _img_to_b64(warped),
            'anno_b64':   _img_to_b64(anno_bgr),
        })

    return detections


# ═══════════════════════════════════════════════════════════════
# 函數 2：批次處理目錄
# ═══════════════════════════════════════════════════════════════

def infer_dir(input_dir: str, output_dir: str,
              model_path: str,
              conf_threshold: float = CONF_THRESHOLD) -> str:
    """
    對 input_dir 下的所有圖片執行推論，結果寫入 output_dir。

    輸出：
      output_dir/results.json   — 全部偵測記錄（含 base64 圖片）
      output_dir/plates/        — 每個車牌投影另存為 <img>_<idx>.jpg

    Returns:
        結果 JSON 路徑
    """
    from ultralytics import YOLO

    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    plates_dir = output_dir / 'plates'
    output_dir.mkdir(parents=True, exist_ok=True)
    plates_dir.mkdir(exist_ok=True)

    # 收集圖片
    exts = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG'}
    img_files = sorted([f for f in input_dir.iterdir() if f.suffix in exts])
    print(f'Found {len(img_files)} images in {input_dir}')

    # 載入模型
    print(f'Loading YOLO model: {model_path}')
    model = YOLO(model_path)
    _init_parseq()

    all_results = []
    t0 = time.time()

    for idx, img_path in enumerate(img_files, 1):
        print(f'[{idx}/{len(img_files)}] {img_path.name}', end=' ', flush=True)
        detections = infer_one(str(img_path), model, conf_threshold)

        for di, det in enumerate(detections):
            # 另存投影車牌圖檔
            plate_fname = f'{img_path.stem}_{di}.jpg'
            plate_fpath = plates_dir / plate_fname
            if det['plate_b64']:
                buf = base64.b64decode(det['plate_b64'])
                plate_fpath.write_bytes(buf)
            det['plate_file'] = str(plate_fpath)

        all_results.extend(detections)
        print(f'→ {len(detections)} plate(s)')

    elapsed = time.time() - t0
    print(f'\n✅ Done: {len(all_results)} plates from {len(img_files)} images '
          f'in {elapsed:.1f}s')

    json_path = output_dir / 'results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f'JSON saved: {json_path}')

    return str(json_path)


# ═══════════════════════════════════════════════════════════════
# 函數 3：產生 HTML 檢視器
# ═══════════════════════════════════════════════════════════════

def generate_html(json_path: str,
                  output_html: Optional[str] = None,
                  per_page: int = 100) -> str:
    """
    從 results.json 產生分頁 HTML 檢視器。

    功能：
      - 每頁 100 格，格內顯示投影車牌 + OCR + 信心
      - 點擊格子：Modal 顯示標注原圖、四角點、偵測信心等詳細資訊
      - 篩選：只顯示低信心（首字或尾字 < 0.99）
      - 排序：依首字信心升序（問題最大的排最前）
      - 頁首統計：總數、高信心數、低信心數

    Returns:
        HTML 檔案路徑
    """
    json_path = Path(json_path)
    if output_html is None:
        output_html = json_path.parent / 'viewer.html'
    output_html = Path(output_html)

    print(f'Loading: {json_path}')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'  {len(data)} records')

    # 統計
    total     = len(data)
    high_conf = sum(1 for r in data
                    if r.get('conf_first', 0) >= 0.99 and r.get('conf_last', 0) >= 0.99)
    low_conf  = total - high_conf

    items_json = json.dumps(data, ensure_ascii=False, separators=(',', ':'))

    html = f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>車牌推論結果檢視器</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0c0c18;color:#ddd;padding:8px}}

/* ── Header ── */
.hdr{{text-align:center;padding:10px 0 4px}}
.hdr h1{{font-size:18px;color:#e94560;letter-spacing:1px;margin-bottom:6px}}
.stats{{display:flex;justify-content:center;gap:20px;flex-wrap:wrap;margin-bottom:6px}}
.stat{{background:#16213e;border:1px solid #252550;border-radius:6px;padding:6px 16px;text-align:center}}
.stat .val{{font-size:20px;font-weight:bold;color:#e94560}}
.stat .lbl{{font-size:10px;color:#777}}

/* ── Filter & Sort bar ── */
.fbar{{
  background:linear-gradient(135deg,#13132a,#16213e);
  border:1px solid #252550;border-radius:8px;padding:8px 14px;
  margin:6px auto;max-width:1400px;
  display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;
}}
.fbar .ft{{color:#e94560;font-weight:bold;font-size:12px}}
.fbar label{{font-size:11px;color:#bbb;cursor:pointer;
  display:flex;align-items:center;gap:3px;user-select:none}}
.fbar label:hover{{color:#fff}}
.fbar input[type=checkbox]{{accent-color:#e94560;width:13px;height:13px}}
.fbar select{{background:#0c0c18;color:#ccc;border:1px solid #333;
  border-radius:4px;padding:2px 6px;font-size:11px}}
.fst{{display:none;border-radius:5px;padding:5px 12px;margin:4px auto;
  max-width:1400px;font-size:11px;text-align:center;
  background:linear-gradient(90deg,#0a1628,#0f2040,#0a1628);
  border:1px solid #1a3060;color:#5dade2}}
.fst.show{{display:block}}

/* ── Pagination ── */
.pg{{display:flex;justify-content:center;gap:3px;margin:6px 0;flex-wrap:wrap}}
.pg button{{background:#13132a;color:#999;border:1px solid #252550;
  padding:3px 9px;cursor:pointer;border-radius:3px;font-size:11px;
  min-width:28px;transition:all .12s}}
.pg button.a{{background:#e94560;color:#fff;border-color:#e94560;font-weight:bold}}
.pg button:hover:not(:disabled){{background:#1a3060;color:#fff}}
.pg button:disabled{{opacity:.25;cursor:default}}

/* ── Grid ── */
.g{{display:grid;grid-template-columns:repeat(10,1fr);gap:3px;margin:4px 0}}
@media(max-width:1100px){{.g{{grid-template-columns:repeat(7,1fr)}}}}
@media(max-width:700px){{.g{{grid-template-columns:repeat(4,1fr)}}}}

.c{{
  background:#12122a;border:1px solid #1c1c38;border-radius:3px;
  cursor:pointer;overflow:hidden;display:flex;flex-direction:column;
  align-items:center;padding:2px 1px;transition:border-color .12s,box-shadow .12s;
}}
.c:hover{{border-color:#e94560;box-shadow:0 0 8px rgba(233,69,96,.25)}}
.c.lo{{border-color:#e94560}}
.c .nm{{font-size:9px;font-weight:700;color:#e94560;text-align:center;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;width:100%;
  padding:1px 2px;line-height:1.4}}
.c .cf{{font-size:7px;color:#555;text-align:center;line-height:1.3}}
.c img{{width:100%;height:auto;display:block;min-height:12px;background:#0a0a16}}

/* ── Modal ── */
.ov{{display:none;position:fixed;top:0;left:0;width:100%;height:100%;
  background:rgba(0,0,0,.92);z-index:1000;justify-content:center;align-items:center}}
.ov.show{{display:flex}}
.ml{{background:#13132e;border:1px solid #2a2a55;border-radius:10px;
  padding:18px 22px;max-width:960px;width:96%;max-height:93vh;
  overflow-y:auto;position:relative}}
.ml .xx{{position:absolute;top:6px;right:12px;font-size:24px;color:#e94560;
  cursor:pointer;line-height:1}}
.ml .xx:hover{{color:#ff6b81}}
.ml h2{{color:#e94560;font-size:17px;margin-bottom:10px}}
.ml table{{width:100%;border-collapse:collapse;margin-bottom:14px}}
.ml th{{color:#e94560;text-align:left;padding:4px 8px;font-size:11px;
  border-bottom:1px solid #1c1c40}}
.ml td{{padding:4px 8px;font-size:11px;border-bottom:1px solid #1c1c40;color:#bbb}}
.ml .ok{{color:#2ecc71}} .ml .ng{{color:#e94560}}
.ml .sec h3{{color:#777;font-size:11px;margin-bottom:4px}}
.ml .sec img{{max-width:100%;border-radius:4px;border:1px solid #2a2a55;margin-bottom:12px}}
.ml .nv{{display:flex;justify-content:space-between;align-items:center;margin-top:12px}}
.ml .nv button{{background:#1a1a40;color:#ccc;border:1px solid #2a2a55;
  padding:5px 14px;border-radius:4px;cursor:pointer;font-size:12px;transition:all .15s}}
.ml .nv button:hover:not(:disabled){{background:#e94560;color:#fff}}
.ml .nv button:disabled{{opacity:.25;cursor:default}}
.ml .nv span{{color:#555;font-size:11px}}
</style>
</head>
<body>

<div class="hdr">
  <h1>🚗 車牌推論結果檢視器</h1>
  <div class="stats">
    <div class="stat"><div class="val">{total}</div><div class="lbl">總計</div></div>
    <div class="stat"><div class="val" style="color:#2ecc71">{high_conf}</div><div class="lbl">高信心 ≥0.99</div></div>
    <div class="stat"><div class="val">{low_conf}</div><div class="lbl">低信心 &lt;0.99</div></div>
  </div>
</div>

<div class="fbar">
  <span class="ft">⚙ 篩選/排序：</span>
  <label><input type="checkbox" id="fLow"> 只顯示低信心（首字或尾字 &lt; 0.99）</label>
  <label>排序：
    <select id="sortBy">
      <option value="none">原始順序</option>
      <option value="cf_asc">首字信心 ↑（最差優先）</option>
      <option value="cf_desc">首字信心 ↓（最好優先）</option>
      <option value="det_desc">偵測信心 ↓</option>
    </select>
  </label>
</div>
<div class="fst" id="fst"></div>

<div class="pg" id="p1"></div>
<div class="g"  id="gd"></div>
<div class="pg" id="p2"></div>

<div class="ov" id="ov" onclick="if(event.target===this)hm()">
  <div class="ml" id="ml"></div>
</div>

<script>
const D={items_json};
const PP={per_page};
let F=[],cp=0,mi=-1;

function $(id){{return document.getElementById(id)}}
function esc(s){{return s?String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;'):'—'}}

function af(){{
  const lo=$('fLow').checked;
  const so=$('sortBy').value;
  F=D.filter(r=>!lo||(r.conf_first<0.99||r.conf_last<0.99));
  if(so==='cf_asc')  F.sort((a,b)=>a.conf_first-b.conf_first);
  if(so==='cf_desc') F.sort((a,b)=>b.conf_first-a.conf_first);
  if(so==='det_desc')F.sort((a,b)=>b.det_conf-a.det_conf);
  const fst=$('fst');
  if(lo||so!=='none'){{
    fst.className='fst show';
    fst.innerHTML='篩選後：<b>'+F.length+'</b> / '+D.length+' 筆';
  }}else{{
    fst.className='fst';
  }}
  cp=0;rp();
}}

function tp(){{return Math.max(1,Math.ceil(F.length/PP))}}
function bph(){{
  const t=tp();
  let h='<button onclick="gp('+(cp-1)+')"'+(cp<1?' disabled':'')+'>◀ 上頁</button>';
  const s=new Set();
  for(let i=0;i<Math.min(3,t);i++) s.add(i);
  for(let i=Math.max(0,t-3);i<t;i++) s.add(i);
  for(let i=Math.max(0,cp-2);i<=Math.min(t-1,cp+2);i++) s.add(i);
  let pv=-1;
  for(const p of[...s].sort((a,b)=>a-b)){{
    if(pv>=0&&p-pv>1) h+='<button disabled>…</button>';
    h+='<button class="'+(p===cp?'a':'')+'" onclick="gp('+p+')">'+(p+1)+'</button>';
    pv=p;
  }}
  h+='<button onclick="gp('+(cp+1)+')"'+(cp>=t-1?' disabled':'')+'>下頁 ▶</button>';
  return h;
}}

function rp(){{
  const h=bph();
  $('p1').innerHTML=h;
  $('p2').innerHTML=h;
  const s=cp*PP,items=F.slice(s,s+PP),el=$('gd');
  el.innerHTML=items.map((r,i)=>{{
    const gi=s+i;
    const lo=(r.conf_first<0.99||r.conf_last<0.99);
    const src=r.plate_b64?'data:image/jpeg;base64,'+r.plate_b64:'';
    return '<div class="c'+(lo?' lo':'')+'" onclick="sm('+gi+')">'+
      '<div class="nm">'+esc(r.ocr_text||'—')+'</div>'+
      (src?'<img src="'+src+'" alt="">':'<div style="height:24px;background:#1a1a2a"></div>')+
      '<div class="cf">1st:'+r.conf_first.toFixed(3)+' last:'+r.conf_last.toFixed(3)+'</div>'+
    '</div>';
  }}).join('');
}}

function gp(p){{
  const t=tp();
  if(p<0||p>=t) return;
  cp=p;rp();window.scrollTo(0,0);
}}

function sm(idx){{
  mi=idx;rmm();$('ov').classList.add('show');
}}

function rmm(){{
  const r=F[mi];
  if(!r) return;
  const lo=(r.conf_first<0.99||r.conf_last<0.99);
  const plateSrc=r.plate_b64?'data:image/jpeg;base64,'+r.plate_b64:'';
  const annoSrc =r.anno_b64 ?'data:image/jpeg;base64,'+r.anno_b64 :'';
  const kps=(r.keypoints||[]).map((p,i)=>'<tr><td>角點'+(i+1)+'</td><td>('+p[0].toFixed(1)+','+p[1].toFixed(1)+')</td><td>'+((r.kp_confs&&r.kp_confs[i])||0).toFixed(3)+'</td></tr>').join('');
  const chars=(r.chars||[]).map(c=>c.char+':'+c.conf.toFixed(3)).join('  ');
  $('ml').innerHTML=
    '<span class="xx" onclick="hm()">&times;</span>'+
    '<h2>'+(mi+1)+' / '+F.length+' — '+esc(r.ocr_text)+'</h2>'+
    '<table>'+
      '<tr><th>項目</th><th>數值</th></tr>'+
      '<tr><td>來源圖片</td><td>'+esc(r.image)+'</td></tr>'+
      '<tr><td>偵測信心</td><td>'+r.det_conf.toFixed(4)+'</td></tr>'+
      '<tr><td>OCR 文字</td><td style="font-weight:bold;color:#2ecc71">'+esc(r.ocr_text)+'</td></tr>'+
      '<tr><td>首字信心</td><td class="'+(r.conf_first>=0.99?'ok':'ng')+'">'+r.conf_first.toFixed(6)+'</td></tr>'+
      '<tr><td>尾字信心</td><td class="'+(r.conf_last >=0.99?'ok':'ng')+'">'+r.conf_last.toFixed(6)+'</td></tr>'+
      '<tr><td>字元明細</td><td style="font-size:10px;color:#aaa">'+esc(chars)+'</td></tr>'+
    '</table>'+
    '<table><tr><th>角點</th><th>座標</th><th>信心</th></tr>'+kps+'</table>'+
    (plateSrc?'<div class="sec"><h3>投影後車牌</h3><img src="'+plateSrc+'"></div>':'')+
    (annoSrc ?'<div class="sec"><h3>原圖標注（框線 + 角點）</h3><img src="'+annoSrc+'"></div>':'')+
    '<div class="nv">'+
      '<button onclick="mn(-1)"'+(mi<=0?' disabled':'')+'>◀ 上一筆</button>'+
      '<span>'+(mi+1)+' / '+F.length+'</span>'+
      '<button onclick="mn(1)"'+(mi>=F.length-1?' disabled':'')+'>下一筆 ▶</button>'+
    '</div>';
}}

function mn(d){{
  const ni=mi+d;
  if(ni<0||ni>=F.length) return;
  mi=ni;rmm();$('ml').scrollTop=0;
}}
function hm(){{$('ov').classList.remove('show');mi=-1}}

document.addEventListener('keydown',e=>{{
  if(e.key==='Escape'){{hm();return}}
  const inM=$('ov').classList.contains('show');
  if(inM){{
    if(e.key==='ArrowLeft'){{e.preventDefault();mn(-1)}}
    if(e.key==='ArrowRight'){{e.preventDefault();mn(1)}}
  }}else{{
    if(e.key==='ArrowLeft') gp(cp-1);
    if(e.key==='ArrowRight') gp(cp+1);
  }}
}});

$('fLow').addEventListener('change',af);
$('sortBy').addEventListener('change',af);
af();
</script>
</body>
</html>'''

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html)
    sz = os.path.getsize(output_html)
    print(f'✅ HTML saved: {output_html} ({sz/1024:.0f} KB)')
    return str(output_html)


# ═══════════════════════════════════════════════════════════════
# 主函數
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='車牌推論 + HTML 檢視器')
    parser.add_argument('--model',      type=str, required=True,
                        help='YOLO 權重路徑，例如 best.pt')
    parser.add_argument('--input',      type=str, required=True,
                        help='輸入圖片目錄')
    parser.add_argument('--output',     type=str, default='./lp_results',
                        help='輸出目錄（預設 ./lp_results）')
    parser.add_argument('--conf',       type=float, default=CONF_THRESHOLD,
                        help=f'偵測信心門檻（預設 {CONF_THRESHOLD}）')
    parser.add_argument('--per-page',   type=int, default=100,
                        help='HTML 每頁格數（預設 100）')
    parser.add_argument('--html-only',  action='store_true',
                        help='跳過推論，只重新產生 HTML（需已有 results.json）')
    args = parser.parse_args()

    output_dir = Path(args.output)
    json_path  = output_dir / 'results.json'
    html_path  = output_dir / 'viewer.html'

    if args.html_only:
        if not json_path.exists():
            print(f'❌ results.json not found: {json_path}')
            sys.exit(1)
        generate_html(str(json_path), str(html_path), per_page=args.per_page)
    else:
        json_path = infer_dir(
            input_dir      = args.input,
            output_dir     = args.output,
            model_path     = args.model,
            conf_threshold = args.conf,
        )
        generate_html(json_path, str(html_path), per_page=args.per_page)

    print(f'\n📁 輸出目錄：{output_dir}')
    print(f'   results.json : {json_path}')
    print(f'   viewer.html  : {html_path}')


if __name__ == '__main__':
    main()
