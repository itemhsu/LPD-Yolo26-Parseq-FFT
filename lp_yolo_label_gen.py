#!/usr/bin/env python3
"""
lp_yolo_label_gen.py — 從 selected_plates JSON + results.json 產生 YOLOv8-Pose 標記檔
==================================================================================
用法：
    python lp_yolo_label_gen.py
    python lp_yolo_label_gen.py --selected selected_plates_429.json --results results.json --output-dir ./yolo_lp/

輸出：
    output-dir/
      labels/          ← YOLO txt 標記檔
      images/          ← 原圖 symlink 或 copy (可選)
      dataset.yaml     ← YOLOv8 dataset config
      label_viewer.html ← 視覺化 HTML

YOLOv8-Pose 格式 (每行):
    class_id  cx  cy  w  h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4
    0 = 車牌
    keypoints: TL(左上) TR(右上) BR(右下) BL(左下), visibility=2(可見)
"""

import os, sys, json, re, math, shutil, argparse
from pathlib import Path

SELECTED_JSON = '/Users/miniaicar/amtk/lp/lp_viewer_tool/lp_viewer_output/selected_plates_429.json'
RESULTS_JSON  = '/Users/miniaicar/amtk/lp/lp_viewer_tool/lp_viewer_output/results.json'
OUTPUT_DIR    = '/Users/miniaicar/amtk/lp/lp_viewer_tool/lp_viewer_output/yolo_lp/'
IMG_BASE      = '/Users/miniaicar/amtk/lp/LPD/'  # original_image 的根目錄

PATH_MAP = {'/home/itemhsu': '/Users/miniaicar'}

def map_path(p):
    for src, dst in PATH_MAP.items():
        if p.startswith(src):
            return p.replace(src, dst, 1)
    return p


def quad_to_yolo_pose(quad, img_w, img_h):
    """
    quad: [[x,y], [x,y], [x,y], [x,y]] — TL, TR, BR, BL
    回傳: (cx, cy, w, h, kp1..kp4) 全部正規化到 [0,1]

    YOLOv8-Pose: class cx cy w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4
    """
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # bbox (normalized)
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h

    # clamp to [0, 1]
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    bw = max(0, min(1, bw))
    bh = max(0, min(1, bh))

    # keypoints: TL TR BR BL, visibility=2 (visible)
    kps = []
    for pt in quad:
        kx = max(0, min(1, pt[0] / img_w))
        ky = max(0, min(1, pt[1] / img_h))
        kps.extend([kx, ky, 2])

    return cx, cy, bw, bh, kps


def generate_labels(selected_path, results_path, output_dir, img_base, use_corrected=True, copy_images=False):
    print(f'Loading selected: {selected_path}')
    with open(selected_path, 'r') as f:
        selected = json.load(f)
    sel_ids = set(r['plate_id'] for r in selected)
    print(f'  {len(sel_ids)} plates')

    print(f'Loading results: {results_path}')
    with open(results_path, 'r') as f:
        results = json.load(f)
    res_by_id = {r['plate_id']: r for r in results if 'plate_id' in r}
    print(f'  {len(res_by_id)} records')

    label_dir = os.path.join(output_dir, 'labels')
    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(label_dir, exist_ok=True)
    if copy_images:
        os.makedirs(image_dir, exist_ok=True)

    # 一張圖可能有多個 plate，按 original_path 分組
    img_plates = {}  # original_path -> list of records
    for pid in sel_ids:
        rec = res_by_id.get(pid)
        if not rec:
            print(f'  [WARN] {pid} not found in results.json')
            continue
        orig_path = map_path(rec.get('original_path', ''))
        if not orig_path:
            continue
        if orig_path not in img_plates:
            img_plates[orig_path] = []
        img_plates[orig_path].append(rec)

    print(f'  {len(img_plates)} unique images')

    # 產生標記
    vis_data = []  # for HTML viewer
    label_count = 0

    for img_path, recs in img_plates.items():
        # image size: [H, W]
        img_size = recs[0].get('image_size', [0, 0])
        img_h, img_w = img_size[0], img_size[1]
        if img_h == 0 or img_w == 0:
            print(f'  [WARN] invalid image_size for {img_path}')
            continue

        # label filename = image filename with .txt
        img_fname = os.path.basename(img_path)
        label_fname = os.path.splitext(img_fname)[0] + '.txt'
        label_path = os.path.join(label_dir, label_fname)

        lines = []
        plate_vis = []
        for rec in recs:
            # 用 corrected_quad 或 keypoints
            if use_corrected and rec.get('corrected_quad'):
                quad = rec['corrected_quad']
            else:
                quad = rec.get('keypoints', [])

            if not quad or len(quad) != 4:
                continue

            cx, cy, bw, bh, kps = quad_to_yolo_pose(quad, img_w, img_h)

            # YOLOv8-Pose 格式
            parts = [f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}']
            for j in range(4):
                parts.append(f'{kps[j*3]:.6f} {kps[j*3+1]:.6f} {int(kps[j*3+2])}')
            lines.append(' '.join(parts))

            plate_vis.append({
                'pid': rec['plate_id'],
                'ocr': rec.get('corrected_ocr', ''),
                'quad': quad,
                'bbox': [cx * img_w, cy * img_h, bw * img_w, bh * img_h],
                'orig_kp': rec.get('keypoints', []),
            })

        if lines:
            with open(label_path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
            label_count += len(lines)

        # copy/symlink image
        if copy_images and os.path.exists(img_path):
            dst = os.path.join(image_dir, img_fname)
            if not os.path.exists(dst):
                shutil.copy2(img_path, dst)

        # resolve relative path for HTML (relative to output_dir)
        # try original_image field for a shorter path
        orig_image = recs[0].get('original_image', '')
        vis_data.append({
            'img': img_fname,
            'img_path': img_path,
            'img_rel': orig_image,
            'w': img_w,
            'h': img_h,
            'plates': plate_vis,
        })

    print(f'  Generated {label_count} labels in {len(img_plates)} files')

    # 產生 dataset.yaml
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    yaml_content = f"""# YOLOv8-Pose License Plate Dataset
# Generated from {os.path.basename(selected_path)}
path: {os.path.abspath(output_dir)}
train: images
val: images

# Keypoint shape: [num_keypoints, dim]
# dim=3: x, y, visibility
kpt_shape: [4, 3]

names:
  0: license_plate

# Keypoint labels (for reference):
# 0: TL (top-left)
# 1: TR (top-right)
# 2: BR (bottom-right)
# 3: BL (bottom-left)
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f'  dataset.yaml saved: {yaml_path}')

    return vis_data, output_dir


def generate_viewer_html(vis_data, output_dir, img_base):
    """產生視覺化 HTML，顯示每張圖的標記"""

    html_path = os.path.join(output_dir, 'label_viewer.html')
    total = len(vis_data)

    # 準備 JS data (不含圖片，用路徑)
    js_items = []
    for v in vis_data:
        plates = []
        for p in v['plates']:
            plates.append({
                'pid': p['pid'],
                'ocr': p['ocr'],
                'q': [[round(x, 2) for x in pt] for pt in p['quad']],
                'oq': [[round(x, 2) for x in pt] for pt in p['orig_kp']] if p['orig_kp'] else [],
            })
        js_items.append({
            'f': v['img'],
            'p': v['img_path'],
            'w': v['w'],
            'h': v['h'],
            'pl': plates,
        })

    items_json = json.dumps(js_items, ensure_ascii=False, separators=(',', ':'))

    html = r'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO Label Viewer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0c0c18;color:#ddd;padding:10px}
.hdr{text-align:center;padding:8px 0}
.hdr h1{font-size:18px;color:#e94560;margin-bottom:4px}
.hdr .st{color:#777;font-size:12px}

.pg{display:flex;justify-content:center;gap:4px;margin:8px 0;flex-wrap:wrap}
.pg button{
  background:#13132a;color:#999;border:1px solid #252550;padding:4px 10px;
  cursor:pointer;border-radius:3px;font-size:12px;min-width:32px;
}
.pg button.a{background:#e94560;color:#fff;border-color:#e94560;font-weight:bold}
.pg button:hover:not(:disabled){background:#1a3060;color:#fff}
.pg button:disabled{opacity:.25;cursor:default}

.canvas-wrap{
  display:flex;justify-content:center;align-items:center;
  margin:10px auto;max-width:1400px;position:relative;
}
canvas{border:1px solid #333;border-radius:4px;max-width:100%;cursor:crosshair}

.info{
  max-width:1400px;margin:8px auto;padding:10px 14px;
  background:#13132a;border:1px solid #252550;border-radius:6px;
}
.info h3{color:#e94560;font-size:13px;margin-bottom:6px}
.info table{width:100%;border-collapse:collapse}
.info th{color:#e94560;text-align:left;padding:3px 8px;font-size:11px;border-bottom:1px solid #1c1c40}
.info td{padding:3px 8px;font-size:11px;border-bottom:1px solid #1c1c40;color:#bbb}

.legend{max-width:1400px;margin:6px auto;font-size:11px;color:#666;text-align:center}
.legend span{margin:0 10px}
.leg-orig{color:#ff6b6b}
.leg-corr{color:#2ecc71}
.leg-bbox{color:#f1c40f}
.leg-kp{color:#3498db}

.ctrl{
  max-width:1400px;margin:6px auto;display:flex;gap:14px;align-items:center;
  justify-content:center;flex-wrap:wrap;
}
.ctrl label{font-size:11px;color:#bbb;cursor:pointer;display:flex;align-items:center;gap:3px}
.ctrl input[type=checkbox]{accent-color:#e94560;width:13px;height:13px}
</style>
</head>
<body>

<div class="hdr">
  <h1>🏷️ YOLO Label 視覺化檢視器</h1>
  <div class="st" id="st"></div>
</div>

<div class="ctrl">
  <label><input type="checkbox" id="showOrig" checked> <span class="leg-orig">■</span> 原始 keypoints</label>
  <label><input type="checkbox" id="showCorr" checked> <span class="leg-corr">■</span> 校正後 quad (YOLO label)</label>
  <label><input type="checkbox" id="showBbox" checked> <span class="leg-bbox">■</span> Bounding box</label>
  <label><input type="checkbox" id="showLabels" checked> 顯示文字</label>
</div>

<div class="pg" id="pg"></div>
<div class="canvas-wrap"><canvas id="cv"></canvas></div>
<div class="info" id="info"></div>
<div class="pg" id="pg2"></div>

<script>
const D=%%DATA%%;
const IMG_BASE='%%IMG_BASE%%';
let cp=0;
const cv=document.getElementById('cv');
const ctx=cv.getContext('2d');

function $(id){return document.getElementById(id)}
function esc(s){return s?s.replace(/&/g,'&amp;').replace(/</g,'&lt;'):''}

$('st').textContent='共 '+D.length+' 張圖 ｜ ← → 切換 ｜ 顯示原始/校正後 keypoints + bbox';

function bph(){
  const t=D.length;
  let h='<button onclick="gp('+(cp-1)+')"'+(cp<1?' disabled':'')+'>◀ 上一張</button>';
  const show=new Set();
  for(let i=0;i<Math.min(3,t);i++) show.add(i);
  for(let i=Math.max(0,t-3);i<t;i++) show.add(i);
  for(let i=Math.max(0,cp-2);i<=Math.min(t-1,cp+2);i++) show.add(i);
  let pv=-1;
  for(const p of[...show].sort((a,b)=>a-b)){
    if(pv>=0&&p-pv>1) h+='<button disabled>…</button>';
    h+='<button class="'+(p===cp?'a':'')+'" onclick="gp('+p+')">'+(p+1)+'</button>';
    pv=p;
  }
  h+='<button onclick="gp('+(cp+1)+')"'+(cp>=t-1?' disabled':'')+'>下一張 ▶</button>';
  h+=' <span style="color:#555;font-size:11px;margin-left:8px">'+(cp+1)+'/'+t+'</span>';
  return h;
}

function render(){
  const ph=bph();
  $('pg').innerHTML=ph;
  $('pg2').innerHTML=ph;

  const d=D[cp];
  if(!d) return;

  const img=new Image();
  img.onload=function(){
    const maxW=Math.min(1380, window.innerWidth-30);
    const scale=Math.min(maxW/d.w, 800/d.h, 1);
    const cw=Math.round(d.w*scale), ch=Math.round(d.h*scale);
    cv.width=cw; cv.height=ch;
    ctx.drawImage(img,0,0,cw,ch);

    const showOrig=$('showOrig').checked;
    const showCorr=$('showCorr').checked;
    const showBbox=$('showBbox').checked;
    const showLabels=$('showLabels').checked;

    for(const p of d.pl){
      // original keypoints (red)
      if(showOrig && p.oq && p.oq.length===4){
        ctx.strokeStyle='rgba(255,107,107,0.8)';
        ctx.lineWidth=2;
        ctx.setLineDash([6,3]);
        ctx.beginPath();
        for(let i=0;i<4;i++){
          const x=p.oq[i][0]*scale, y=p.oq[i][1]*scale;
          if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        }
        ctx.closePath();
        ctx.stroke();
        ctx.setLineDash([]);

        // keypoint dots
        const kpLabels=['TL','TR','BR','BL'];
        for(let i=0;i<4;i++){
          const x=p.oq[i][0]*scale, y=p.oq[i][1]*scale;
          ctx.fillStyle='#ff6b6b';
          ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill();
          if(showLabels){
            ctx.fillStyle='#ff6b6b';ctx.font='9px sans-serif';
            ctx.fillText(kpLabels[i],x+5,y-4);
          }
        }
      }

      // corrected quad (green) - this is what YOLO label uses
      if(showCorr && p.q && p.q.length===4){
        ctx.strokeStyle='rgba(46,204,113,0.9)';
        ctx.lineWidth=2;
        ctx.setLineDash([]);
        ctx.beginPath();
        for(let i=0;i<4;i++){
          const x=p.q[i][0]*scale, y=p.q[i][1]*scale;
          if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        }
        ctx.closePath();
        ctx.stroke();

        // keypoint circles
        const kpLabels=['TL','TR','BR','BL'];
        const kpColors=['#3498db','#e74c3c','#f39c12','#9b59b6'];
        for(let i=0;i<4;i++){
          const x=p.q[i][0]*scale, y=p.q[i][1]*scale;
          ctx.fillStyle=kpColors[i];
          ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill();
          ctx.strokeStyle='#fff';ctx.lineWidth=1;
          ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.stroke();
          if(showLabels){
            ctx.fillStyle='#fff';ctx.font='bold 10px sans-serif';
            ctx.fillText(kpLabels[i],x+7,y+3);
          }
        }
      }

      // bbox (yellow dashed)
      if(showBbox && p.q && p.q.length===4){
        const xs=p.q.map(v=>v[0]*scale), ys=p.q.map(v=>v[1]*scale);
        const x0=Math.min(...xs), x1=Math.max(...xs);
        const y0=Math.min(...ys), y1=Math.max(...ys);
        ctx.strokeStyle='rgba(241,196,15,0.7)';
        ctx.lineWidth=1;
        ctx.setLineDash([4,4]);
        ctx.strokeRect(x0,y0,x1-x0,y1-y0);
        ctx.setLineDash([]);
      }

      // OCR label
      if(showLabels && p.ocr){
        const xs=p.q.map(v=>v[0]*scale);
        const ys=p.q.map(v=>v[1]*scale);
        const lx=Math.min(...xs), ly=Math.min(...ys)-6;
        ctx.fillStyle='rgba(0,0,0,0.7)';
        ctx.fillRect(lx,ly-12,ctx.measureText(p.ocr).width+8,14);
        ctx.fillStyle='#2ecc71';ctx.font='bold 11px sans-serif';
        ctx.fillText(p.ocr,lx+4,ly);
      }
    }
  };
  img.onerror=function(){
    cv.width=600;cv.height=100;
    ctx.fillStyle='#1a1a2a';ctx.fillRect(0,0,600,100);
    ctx.fillStyle='#e94560';ctx.font='14px sans-serif';
    ctx.fillText('Image not found: '+d.f,20,55);
  };
  img.src=d.p;

  // info table
  let tbl='<h3>'+esc(d.f)+' ('+d.w+'×'+d.h+') — '+d.pl.length+' plate(s)</h3>';
  tbl+='<table><tr><th>Plate ID</th><th>OCR</th><th>TL</th><th>TR</th><th>BR</th><th>BL</th></tr>';
  for(const p of d.pl){
    const q=p.q||[];
    tbl+='<tr><td>'+esc(p.pid)+'</td><td style="color:#2ecc71;font-weight:bold">'+esc(p.ocr)+'</td>';
    for(let i=0;i<4;i++){
      if(q[i]) tbl+='<td>('+q[i][0].toFixed(1)+', '+q[i][1].toFixed(1)+')</td>';
      else tbl+='<td>—</td>';
    }
    tbl+='</tr>';
  }
  tbl+='</table>';
  $('info').innerHTML=tbl;
}

function gp(p){
  if(p<0||p>=D.length)return;
  cp=p;
  render();
  window.scrollTo(0,0);
}

document.addEventListener('keydown',e=>{
  if(e.key==='ArrowLeft') gp(cp-1);
  if(e.key==='ArrowRight') gp(cp+1);
});

['showOrig','showCorr','showBbox','showLabels'].forEach(id=>{
  $(id).addEventListener('change',render);
});

render();
</script>
</body>
</html>'''

    html = html.replace('%%DATA%%', items_json)
    html = html.replace('%%IMG_BASE%%', img_base)

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    sz = os.path.getsize(html_path)
    print(f'  HTML viewer saved: {html_path} ({sz/1024:.0f} KB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='產生 YOLOv8-Pose 車牌標記檔 + 視覺化 HTML')
    parser.add_argument('--selected', type=str, default=SELECTED_JSON,
                        help='selected plates JSON 路徑')
    parser.add_argument('--results', type=str, default=RESULTS_JSON,
                        help='results.json 路徑 (含 keypoints, image_size)')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='輸出目錄')
    parser.add_argument('--img-base', type=str, default=IMG_BASE,
                        help='原始圖片根目錄 (original_image 的前綴)')
    parser.add_argument('--use-original-kp', action='store_true',
                        help='使用原始 keypoints 而非 corrected_quad')
    parser.add_argument('--copy-images', action='store_true',
                        help='複製原圖到 output-dir/images/')
    args = parser.parse_args()

    vis_data, out_dir = generate_labels(
        args.selected, args.results, args.output_dir,
        args.img_base,
        use_corrected=not args.use_original_kp,
        copy_images=args.copy_images,
    )
    generate_viewer_html(vis_data, out_dir, args.img_base)
