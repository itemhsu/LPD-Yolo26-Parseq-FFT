#!/usr/bin/env python3
"""
lp_review_dataset.py — 審查合併後的 YOLOv8-Pose 資料集 zip
==========================================================
用法：
    # Step 1: 解壓 + 驗證 + 產生 HTML 審查器
    python lp_review_dataset.py --zip merged_lp_dataset.zip

    # Step 2: 在 HTML 中標記要刪除的，匯出 delete_list.json

    # Step 3: 依據 delete_list.json 清理並打包
    python lp_review_dataset.py --zip merged_lp_dataset.zip --delete delete_list.json --output cleaned_dataset.zip

驗證項目：
    - 圖片無對應 label / label 無對應圖片
    - label 格式錯誤（欄位數不對、超出 [0,1] 範圍）
    - bbox 過小 / 過大
    - keypoint 順序異常（非凸四邊形、交叉）
    - 圖片損壞（無法讀取）
    - 重複圖片（依檔案 hash）
"""

import os, sys, json, zipfile, shutil, hashlib, argparse, re
from pathlib import Path
from collections import defaultdict

WORK_DIR = '_review_work'
EXPECTED_FIELDS = 17   # YOLOv8-Pose: 1 class + 4 bbox + 4*(x,y,vis)


def extract_zip(zip_path, work_dir):
    print(f'Extracting: {zip_path}')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(work_dir)
    return work_dir


def validate_label(label_path):
    """驗證單個 label 檔，回傳 (issues, parsed_boxes)"""
    issues = []
    boxes = []
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        return [f'Cannot read: {e}'], []

    if not lines or all(l.strip() == '' for l in lines):
        issues.append('Empty label file')
        return issues, []

    for li, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        parts = line.split()

        # Field count
        if len(parts) != EXPECTED_FIELDS:
            issues.append(f'Line {li}: expected {EXPECTED_FIELDS} fields, got {len(parts)}')
            continue

        try:
            vals = [float(v) for v in parts]
        except ValueError:
            issues.append(f'Line {li}: non-numeric values')
            continue

        cls = int(vals[0])
        cx, cy, bw, bh = vals[1], vals[2], vals[3], vals[4]

        # bbox range
        for name, val in [('cx', cx), ('cy', cy), ('bw', bw), ('bh', bh)]:
            if val < 0 or val > 1:
                issues.append(f'Line {li}: {name}={val:.4f} out of [0,1]')

        # bbox size
        if bw < 0.005 or bh < 0.005:
            issues.append(f'Line {li}: bbox too small ({bw:.4f}x{bh:.4f})')
        if bw > 0.95 or bh > 0.95:
            issues.append(f'Line {li}: bbox too large ({bw:.4f}x{bh:.4f})')

        # keypoints
        kps = []
        for ki in range(4):
            kx = vals[5 + ki * 3]
            ky = vals[6 + ki * 3]
            kv = int(vals[7 + ki * 3])
            if kx < 0 or kx > 1 or ky < 0 or ky > 1:
                issues.append(f'Line {li}: kp{ki} ({kx:.4f},{ky:.4f}) out of [0,1]')
            kps.append((kx, ky, kv))

        # check convexity (cross product sign should be consistent)
        if len(kps) == 4:
            pts = [(kp[0], kp[1]) for kp in kps]
            crosses = []
            for i in range(4):
                p0 = pts[i]
                p1 = pts[(i + 1) % 4]
                p2 = pts[(i + 2) % 4]
                cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
                crosses.append(cross)
            signs = [1 if c > 0 else (-1 if c < 0 else 0) for c in crosses]
            non_zero = [s for s in signs if s != 0]
            if non_zero and not (all(s > 0 for s in non_zero) or all(s < 0 for s in non_zero)):
                issues.append(f'Line {li}: keypoints form non-convex/crossed quad')

        boxes.append({
            'cls': cls, 'cx': cx, 'cy': cy, 'bw': bw, 'bh': bh,
            'kps': kps
        })

    return issues, boxes


def check_image(img_path):
    """快速檢查圖片是否可讀"""
    try:
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            return 'cv2.imread returned None'
        h, w = img.shape[:2]
        if h < 10 or w < 10:
            return f'Image too small: {w}x{h}'
        return None
    except ImportError:
        # fallback: just check file size
        sz = os.path.getsize(img_path)
        if sz < 100:
            return f'File too small: {sz} bytes'
        return None
    except Exception as e:
        return str(e)


def file_hash(path, algo='md5'):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def review_dataset(work_dir):
    """全面審查資料集，回傳 review 結果"""
    results = {'splits': {}, 'issues': [], 'items': []}

    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(work_dir, split, 'images')
        lbl_dir = os.path.join(work_dir, split, 'labels')

        if not os.path.exists(img_dir):
            continue

        imgs = set()
        lbls = set()
        if os.path.exists(img_dir):
            imgs = {os.path.splitext(f)[0] for f in os.listdir(img_dir)
                    if f.endswith(('.jpg', '.jpeg', '.png'))}
        if os.path.exists(lbl_dir):
            lbls = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir)
                    if f.endswith('.txt')}

        # orphan checks
        imgs_no_lbl = imgs - lbls
        lbls_no_img = lbls - imgs
        matched = imgs & lbls

        results['splits'][split] = {
            'images': len(imgs),
            'labels': len(lbls),
            'matched': len(matched),
            'img_no_label': len(imgs_no_lbl),
            'label_no_img': len(lbls_no_img),
        }

        for name in sorted(imgs_no_lbl):
            results['issues'].append({
                'split': split, 'name': name, 'type': 'img_no_label',
                'msg': 'Image has no matching label file'
            })

        for name in sorted(lbls_no_img):
            results['issues'].append({
                'split': split, 'name': name, 'type': 'label_no_img',
                'msg': 'Label has no matching image file'
            })

        # Validate each matched pair
        hash_map = {}  # hash -> (split, name) for duplicate detection
        for name in sorted(matched):
            item = {'split': split, 'name': name, 'issues': [], 'boxes': []}

            # find actual image filename
            img_file = None
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = os.path.join(img_dir, name + ext)
                if os.path.exists(candidate):
                    img_file = candidate
                    break

            lbl_file = os.path.join(lbl_dir, name + '.txt')

            # check image
            if img_file:
                img_issue = check_image(img_file)
                if img_issue:
                    item['issues'].append(f'Image: {img_issue}')

                # duplicate check
                h = file_hash(img_file)
                if h in hash_map:
                    dup = hash_map[h]
                    item['issues'].append(f'Duplicate of {dup[0]}/{dup[1]}')
                else:
                    hash_map[h] = (split, name)

                # get image dimensions for display
                try:
                    import cv2
                    im = cv2.imread(img_file)
                    if im is not None:
                        item['img_h'], item['img_w'] = im.shape[:2]
                except:
                    pass

                item['img_path'] = os.path.relpath(img_file, work_dir)
            else:
                item['issues'].append('Image file not found')

            # check label
            lbl_issues, boxes = validate_label(lbl_file)
            item['issues'].extend(lbl_issues)
            item['boxes'] = boxes
            item['lbl_path'] = os.path.relpath(lbl_file, work_dir)

            results['items'].append(item)

    return results


def generate_review_html(results, work_dir, output_html):
    """產生互動式審查 HTML"""
    items = results['items']

    # 分類
    error_items = [i for i, it in enumerate(items) if it['issues']]
    clean_items = [i for i, it in enumerate(items) if not it['issues']]

    # 準備 JS 資料（不含大圖，用路徑）
    js_items = []
    for it in items:
        js_items.append({
            's': it['split'],
            'n': it['name'],
            'is': it['issues'],
            'b': len(it.get('boxes', [])),
            'ip': it.get('img_path', ''),
            'w': it.get('img_w', 0),
            'h': it.get('img_h', 0),
            'bx': it.get('boxes', []),
        })

    data_json = json.dumps(js_items, ensure_ascii=False, separators=(',', ':'))
    split_stats = json.dumps(results['splits'], indent=2)

    html = r'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dataset Review</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;500;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Outfit',sans-serif;background:#0a0a12;color:#ddd;padding:12px}

.hdr{text-align:center;padding:8px 0}
.hdr h1{font-size:22px;color:#ff6b6b;font-weight:700;letter-spacing:2px}
.hdr .sub{color:#666;font-size:12px;margin-top:4px;font-family:'JetBrains Mono',monospace}

/* Stats */
.stats{display:flex;gap:12px;justify-content:center;margin:12px auto;max-width:1400px;flex-wrap:wrap}
.stat-card{background:#12121e;border:1px solid #1e1e35;border-radius:8px;padding:12px 20px;text-align:center;min-width:140px}
.stat-card .num{font-size:28px;font-weight:700;font-family:'JetBrains Mono',monospace}
.stat-card .label{font-size:11px;color:#888;margin-top:2px}
.stat-card.err .num{color:#ff6b6b}
.stat-card.ok .num{color:#2ecc71}
.stat-card.warn .num{color:#f39c12}

/* Toolbar */
.toolbar{background:#12121e;border:1px solid #1e1e35;border-radius:8px;padding:10px 16px;
  margin:10px auto;max-width:1400px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.toolbar .ft{color:#ff6b6b;font-weight:700;font-size:13px}
.toolbar label{font-size:12px;color:#bbb;cursor:pointer;display:flex;align-items:center;gap:4px}
.toolbar input[type=checkbox]{accent-color:#ff6b6b;width:14px;height:14px}
.toolbar button{background:#1a1a30;color:#ccc;border:1px solid #2a2a50;padding:5px 14px;
  border-radius:5px;cursor:pointer;font-size:12px;font-family:'Outfit',sans-serif;transition:all .15s}
.toolbar button:hover{background:#2a2a50;color:#fff}
.toolbar button.danger{background:#6b1a1a;border-color:#ff6b6b;color:#ff6b6b}
.toolbar button.danger:hover{background:#ff6b6b;color:#fff}
.toolbar button.export{background:#1a4020;border-color:#2ecc71;color:#2ecc71}
.toolbar button.export:hover{background:#2ecc71;color:#000}
.toolbar .count{color:#888;font-size:11px;font-family:'JetBrains Mono',monospace}

/* Filter stats */
.fst{display:none;background:#1a1020;border:1px solid #301040;border-radius:6px;
  padding:6px 14px;margin:6px auto;max-width:1400px;font-size:11px;color:#c084fc;text-align:center}
.fst.show{display:block}
.fst b{color:#ff6b6b}

/* Pagination */
.pg{display:flex;justify-content:center;gap:3px;margin:8px 0;flex-wrap:wrap}
.pg button{background:#12121e;color:#888;border:1px solid #1e1e35;padding:3px 10px;
  cursor:pointer;border-radius:3px;font-size:11px;font-family:'JetBrains Mono',monospace}
.pg button.a{background:#ff6b6b;color:#fff;border-color:#ff6b6b;font-weight:700}
.pg button:hover:not(:disabled){background:#1e1e35;color:#fff}
.pg button:disabled{opacity:.2;cursor:default}

/* Grid 10x10 */
.g{display:grid;grid-template-columns:repeat(10,1fr);gap:3px;margin:6px 0}
@media(max-width:1100px){.g{grid-template-columns:repeat(7,1fr)}}
@media(max-width:700px){.g{grid-template-columns:repeat(4,1fr)}}

.c{background:#12121e;border:1px solid #1e1e35;border-radius:4px;cursor:pointer;
  overflow:hidden;display:flex;flex-direction:column;align-items:center;padding:2px;
  position:relative;transition:border-color .12s,opacity .2s}
.c:hover{border-color:#ff6b6b}
.c.has-issue{border-color:#6b3030}
.c.marked{opacity:.25;border-color:#ff0000}
.c .nm{font-size:8px;font-weight:500;color:#888;text-align:center;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;width:100%;padding:1px 2px}
.c .split-tag{font-size:7px;padding:1px 4px;border-radius:2px;color:#fff;position:absolute;top:2px;right:2px}
.c .split-tag.train{background:#2563eb}
.c .split-tag.valid{background:#7c3aed}
.c .split-tag.test{background:#059669}
.c .err-dot{position:absolute;top:2px;left:2px;width:8px;height:8px;border-radius:50%;background:#ff6b6b}
.c img{width:100%;height:auto;display:block;min-height:12px;background:#0a0a12;transition:opacity .3s}
.c img.ld{opacity:.1}
.c .ck{position:absolute;bottom:2px;left:2px;accent-color:#ff6b6b;width:13px;height:13px;cursor:pointer;z-index:3}

/* Modal */
.ov{display:none;position:fixed;top:0;left:0;width:100%;height:100%;
  background:rgba(0,0,0,.94);z-index:1000;justify-content:center;align-items:center}
.ov.show{display:flex}
.ml{background:#0f0f1e;border:1px solid #2a2a55;border-radius:12px;padding:20px 24px;
  max-width:950px;width:96%;max-height:94vh;overflow-y:auto;position:relative}
.ml .xx{position:absolute;top:8px;right:14px;font-size:24px;color:#ff6b6b;cursor:pointer}
.ml h2{color:#ff6b6b;font-size:18px;margin-bottom:12px;font-family:'JetBrains Mono',monospace}
.ml .issue-list{background:#1a0a0a;border:1px solid #400a0a;border-radius:6px;padding:10px;margin-bottom:14px}
.ml .issue-list .issue{color:#ff6b6b;font-size:12px;padding:2px 0;font-family:'JetBrains Mono',monospace}
.ml .issue-list .issue::before{content:'⚠ ';color:#f39c12}
.ml .clean-badge{background:#0a2010;border:1px solid #1a4020;border-radius:6px;padding:8px;
  color:#2ecc71;font-size:12px;margin-bottom:14px;text-align:center}
.ml .canvas-wrap{margin:12px 0;text-align:center}
.ml canvas{border:1px solid #1e1e35;border-radius:6px;max-width:100%}
.ml table{width:100%;border-collapse:collapse;margin:12px 0}
.ml th{color:#ff6b6b;text-align:left;padding:4px 8px;font-size:11px;border-bottom:1px solid #1e1e35}
.ml td{padding:4px 8px;font-size:11px;border-bottom:1px solid #1e1e35;color:#bbb;font-family:'JetBrains Mono',monospace}
.ml .nv{display:flex;justify-content:space-between;align-items:center;margin-top:14px}
.ml .nv button{background:#1a1a30;color:#ccc;border:1px solid #2a2a50;padding:5px 14px;
  border-radius:4px;cursor:pointer;font-size:12px}
.ml .nv button:hover:not(:disabled){background:#ff6b6b;color:#fff}
.ml .nv button:disabled{opacity:.2}
.ml .nv span{color:#555;font-size:11px;font-family:'JetBrains Mono',monospace}
</style>
</head>
<body>

<div class="hdr">
  <h1>🔍 DATASET REVIEW</h1>
  <div class="sub" id="sub"></div>
</div>

<div class="stats" id="stats"></div>

<div class="toolbar">
  <span class="ft">View:</span>
  <label><input type="checkbox" id="fErr" checked> Issues only</label>
  <label><input type="checkbox" id="fTrain" checked> train</label>
  <label><input type="checkbox" id="fValid" checked> valid</label>
  <label><input type="checkbox" id="fTest" checked> test</label>
  <span style="flex:1"></span>
  <span class="count" id="selCnt"></span>
  <button onclick="markAllIssues()">Mark all w/ issues</button>
  <button onclick="markPage()">Mark page</button>
  <button onclick="unmarkPage()">Unmark page</button>
  <button class="danger" onclick="exportDelete()">🗑 Export delete list</button>
  <button class="export" onclick="exportClean()">✅ Export clean list</button>
</div>
<div class="fst" id="fst"></div>

<div class="pg" id="p1"></div>
<div class="g" id="gd"></div>
<div class="pg" id="p2"></div>

<div class="ov" id="ov" onclick="if(event.target===this)hm()">
  <div class="ml" id="ml"></div>
</div>

<script>
const D=%%DATA%%;
const PP=100;
let F=[],cp=0,mi=-1;
const MK={};  // marked for deletion: name->true

function $(id){return document.getElementById(id)}
function esc(s){return s?s.replace(/&/g,'&amp;').replace(/</g,'&lt;'):''}

// Stats
const totalErr=D.filter(r=>r.is.length>0).length;
const totalClean=D.length-totalErr;
$('sub').textContent=D.length+' items | '+totalErr+' with issues | '+totalClean+' clean';
$('stats').innerHTML=
  '<div class="stat-card"><div class="num">'+D.length+'</div><div class="label">Total</div></div>'+
  '<div class="stat-card err"><div class="num">'+totalErr+'</div><div class="label">Issues</div></div>'+
  '<div class="stat-card ok"><div class="num">'+totalClean+'</div><div class="label">Clean</div></div>'+
  %%SPLIT_STATS_HTML%%;

// Filter
function af(){
  const errOnly=$('fErr').checked;
  const showTrain=$('fTrain').checked, showValid=$('fValid').checked, showTest=$('fTest').checked;
  F=D.filter((r,i)=>{
    if(errOnly && r.is.length===0) return false;
    if(!showTrain && r.s==='train') return false;
    if(!showValid && r.s==='valid') return false;
    if(!showTest && r.s==='test') return false;
    r._idx=i;
    return true;
  });
  const el=$('fst');
  el.textContent='Showing '+F.length+' / '+D.length+' items';
  el.classList.add('show');
  cp=0;rp();
}

// Pagination
function tp(){return Math.max(1,Math.ceil(F.length/PP))}
function bph(){
  const t=tp();
  let h='<button onclick="gp('+(cp-1)+')"'+(cp<1?' disabled':'')+'>◀</button>';
  const s=new Set();
  for(let i=0;i<Math.min(3,t);i++)s.add(i);
  for(let i=Math.max(0,t-3);i<t;i++)s.add(i);
  for(let i=Math.max(0,cp-2);i<=Math.min(t-1,cp+2);i++)s.add(i);
  let pv=-1;
  for(const p of[...s].sort((a,b)=>a-b)){
    if(pv>=0&&p-pv>1)h+='<button disabled>…</button>';
    h+='<button class="'+(p===cp?'a':'')+'" onclick="gp('+p+')">'+(p+1)+'</button>';pv=p;
  }
  h+='<button onclick="gp('+(cp+1)+')"'+(cp>=t-1?' disabled':'')+'>▶</button>';
  return h;
}

function updMkCnt(){
  const cnt=Object.values(MK).filter(Boolean).length;
  $('selCnt').textContent=cnt+' marked for deletion';
}

// Render
function rp(){
  const h=bph();$('p1').innerHTML=h;$('p2').innerHTML=h;
  const s=cp*PP,items=F.slice(s,s+PP),el=$('gd');
  el.innerHTML=items.map((r,i)=>{
    const gi=s+i;
    const hasIssue=r.is.length>0;
    const marked=MK[r.s+'/'+r.n];
    return '<div class="c'+(hasIssue?' has-issue':'')+(marked?' marked':'')+'" onclick="sm('+gi+')">'+
      '<span class="split-tag '+r.s+'">'+r.s+'</span>'+
      (hasIssue?'<span class="err-dot"></span>':'')+
      '<div class="nm">'+esc(r.n.substring(0,20))+'</div>'+
      '<img data-g="'+gi+'" class="ld" alt="">'+
      '<input type="checkbox" class="ck"'+(marked?' checked':'')+
        ' onclick="tck(event,\''+r.s+'/'+r.n+'\')" title="Mark for deletion">'+
    '</div>';
  }).join('');

  requestAnimationFrame(()=>{
    el.querySelectorAll('img[data-g]').forEach(img=>{
      const idx=parseInt(img.dataset.g);
      const r=F[idx];if(!r)return;
      if(r.ip){img.src=r.ip;img.onload=function(){this.classList.remove('ld')};
        img.onerror=function(){this.style.background='#2a0a0a';this.classList.remove('ld')}}
      else{img.classList.remove('ld');img.style.background='#1a1a2a'}
    });
  });
  updMkCnt();
}

function gp(p){if(p<0||p>=tp())return;cp=p;rp();window.scrollTo(0,0)}

// Checkbox
function tck(ev,key){
  ev.stopPropagation();
  MK[key]=ev.target.checked;
  const card=ev.target.closest('.c');
  if(card){ev.target.checked?card.classList.add('marked'):card.classList.remove('marked')}
  updMkCnt();
}

function markAllIssues(){
  D.forEach(r=>{if(r.is.length>0)MK[r.s+'/'+r.n]=true});rp();
}
function markPage(){
  const s=cp*PP;F.slice(s,s+PP).forEach(r=>{MK[r.s+'/'+r.n]=true});rp();
}
function unmarkPage(){
  const s=cp*PP;F.slice(s,s+PP).forEach(r=>{MK[r.s+'/'+r.n]=false});rp();
}

// Export
function exportDelete(){
  const list=Object.entries(MK).filter(([k,v])=>v).map(([k])=>{
    const [s,...rest]=k.split('/');return{split:s,name:rest.join('/')};
  });
  if(!list.length){alert('No items marked');return}
  const blob=new Blob([JSON.stringify(list,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='delete_list_'+list.length+'.json';a.click();
}
function exportClean(){
  const marked=new Set(Object.entries(MK).filter(([k,v])=>v).map(([k])=>k));
  const clean=D.filter(r=>!marked.has(r.s+'/'+r.n)).map(r=>({split:r.s,name:r.n}));
  const blob=new Blob([JSON.stringify(clean,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='clean_list_'+clean.length+'.json';a.click();
}

// Modal
function sm(idx){mi=idx;rmm();$('ov').classList.add('show')}
function rmm(){
  const r=F[mi];if(!r)return;
  const ml=$('ml');
  let issueHtml='';
  if(r.is.length>0){
    issueHtml='<div class="issue-list">'+r.is.map(i=>'<div class="issue">'+esc(i)+'</div>').join('')+'</div>';
  }else{
    issueHtml='<div class="clean-badge">✅ No issues detected</div>';
  }
  ml.innerHTML=
    '<span class="xx" onclick="hm()">&times;</span>'+
    '<h2>'+esc(r.n)+'</h2>'+
    '<table><tr><th>Split</th><td>'+r.s+'</td><th>Boxes</th><td>'+r.b+'</td><th>Size</th><td>'+(r.w||'?')+'×'+(r.h||'?')+'</td></tr></table>'+
    issueHtml+
    '<div class="canvas-wrap"><canvas id="cv"></canvas></div>'+
    (r.bx.length?'<table><tr><th>#</th><th>Class</th><th>cx</th><th>cy</th><th>w</th><th>h</th><th>TL</th><th>TR</th><th>BR</th><th>BL</th></tr>'+
      r.bx.map((b,i)=>'<tr><td>'+i+'</td><td>'+b.cls+'</td><td>'+b.cx.toFixed(4)+'</td><td>'+b.cy.toFixed(4)+'</td><td>'+b.bw.toFixed(4)+'</td><td>'+b.bh.toFixed(4)+'</td>'+
        b.kps.map(k=>'<td>('+k[0].toFixed(3)+','+k[1].toFixed(3)+')</td>').join('')+'</tr>').join('')+'</table>':'')+
    '<div class="nv">'+
      '<button onclick="mn(-1)"'+(mi<=0?' disabled':'')+'> ◀ Prev</button>'+
      '<span>'+(mi+1)+' / '+F.length+'</span>'+
      '<button onclick="mn(1)"'+(mi>=F.length-1?' disabled':'')+'> Next ▶</button>'+
    '</div>';

  // Draw on canvas
  if(r.ip && r.w && r.h){
    const cv=document.getElementById('cv');
    const ctx=cv.getContext('2d');
    const img=new Image();
    img.onload=function(){
      const maxW=Math.min(880,window.innerWidth-60);
      const sc=Math.min(maxW/r.w,600/r.h,1);
      cv.width=Math.round(r.w*sc);cv.height=Math.round(r.h*sc);
      ctx.drawImage(img,0,0,cv.width,cv.height);
      const colors=['#2ecc71','#3498db','#e74c3c','#f39c12'];
      for(const b of r.bx){
        // bbox
        const bx=(b.cx-b.bw/2)*cv.width, by=(b.cy-b.bh/2)*cv.height;
        const bww=b.bw*cv.width, bhh=b.bh*cv.height;
        ctx.strokeStyle='rgba(241,196,15,0.6)';ctx.lineWidth=1;ctx.setLineDash([4,4]);
        ctx.strokeRect(bx,by,bww,bhh);ctx.setLineDash([]);
        // quad
        ctx.strokeStyle='#2ecc71';ctx.lineWidth=2;ctx.beginPath();
        const labels=['TL','TR','BR','BL'];
        for(let i=0;i<4;i++){
          const x=b.kps[i][0]*cv.width, y=b.kps[i][1]*cv.height;
          if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
        }
        ctx.closePath();ctx.stroke();
        for(let i=0;i<4;i++){
          const x=b.kps[i][0]*cv.width, y=b.kps[i][1]*cv.height;
          ctx.fillStyle=colors[i];ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill();
          ctx.fillStyle='#fff';ctx.font='bold 9px JetBrains Mono,monospace';ctx.fillText(labels[i],x+5,y-3);
        }
      }
    };
    img.src=r.ip;
  }
}
function mn(d){const ni=mi+d;if(ni<0||ni>=F.length)return;mi=ni;rmm();$('ml').scrollTop=0}
function hm(){$('ov').classList.remove('show');mi=-1}

document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){hm();return}
  const inModal=$('ov').classList.contains('show');
  if(inModal){if(e.key==='ArrowLeft'){e.preventDefault();mn(-1)}if(e.key==='ArrowRight'){e.preventDefault();mn(1)}}
  else{if(e.key==='ArrowLeft')gp(cp-1);if(e.key==='ArrowRight')gp(cp+1)}
});

['fErr','fTrain','fValid','fTest'].forEach(id=>{$(id).addEventListener('change',af)});
af();
</script>
</body>
</html>'''

    # Build split stats HTML for the stat cards
    splits = results['splits']
    split_html_parts = []
    for s in ['train', 'valid', 'test']:
        if s in splits:
            sp = splits[s]
            color = 'warn' if sp['img_no_label'] > 0 or sp['label_no_img'] > 0 else 'ok'
            split_html_parts.append(
                f"'<div class=\"stat-card {color}\"><div class=\"num\">{sp['images']}</div>"
                f"<div class=\"label\">{s}</div></div>'"
            )
    split_stats_html = '+'.join(split_html_parts) if split_html_parts else "''"

    html = html.replace('%%DATA%%', data_json)
    html = html.replace('%%SPLIT_STATS_HTML%%', split_stats_html)

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'HTML viewer: {output_html} ({os.path.getsize(output_html)/1024:.0f} KB)')


def apply_delete_list(zip_path, delete_json, output_zip):
    """依據 delete_list.json 從 zip 中移除檔案並重新打包"""
    print(f'Loading delete list: {delete_json}')
    with open(delete_json, 'r') as f:
        delete_list = json.load(f)
    print(f'  {len(delete_list)} items to delete')

    # Build set of paths to skip
    skip = set()
    for item in delete_list:
        s = item['split']
        n = item['name']
        skip.add(f'{s}/images/{n}.jpg')
        skip.add(f'{s}/images/{n}.jpeg')
        skip.add(f'{s}/images/{n}.png')
        skip.add(f'{s}/labels/{n}.txt')

    print(f'Creating: {output_zip}')
    removed = 0
    kept = 0
    with zipfile.ZipFile(zip_path, 'r') as zin:
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                if info.filename in skip:
                    removed += 1
                    continue
                data = zin.read(info.filename)
                zout.writestr(info, data)
                kept += 1

    sz = os.path.getsize(output_zip)
    print(f'  Removed: {removed} files, Kept: {kept} files')
    print(f'  Output: {output_zip} ({sz/1024/1024:.1f} MB)')


def main():
    parser = argparse.ArgumentParser(description='審查 YOLOv8-Pose 資料集')
    parser.add_argument('--zip', type=str, required=True, help='資料集 zip')
    parser.add_argument('--delete', type=str, default=None,
                        help='delete_list.json (Step 2: 清理用)')
    parser.add_argument('--output', type=str, default=None,
                        help='清理後的 zip (配合 --delete 使用)')
    parser.add_argument('--work-dir', type=str, default=WORK_DIR)
    args = parser.parse_args()

    if args.delete:
        # Step 2: apply delete list
        output = args.output or args.zip.replace('.zip', '_cleaned.zip')
        apply_delete_list(args.zip, args.delete, output)
    else:
        # Step 1: extract + validate + generate HTML
        work_dir = extract_zip(args.zip, args.work_dir)
        print('\nValidating...')
        results = review_dataset(work_dir)

        # Print summary
        print(f'\n{"="*50}')
        print('REVIEW SUMMARY')
        print(f'{"="*50}')
        for split, st in results['splits'].items():
            print(f'  {split}: {st["images"]} imgs, {st["labels"]} lbls, '
                  f'{st["img_no_label"]} orphan imgs, {st["label_no_img"]} orphan lbls')

        n_issues = sum(1 for it in results['items'] if it['issues'])
        print(f'\n  Total items: {len(results["items"])}')
        print(f'  With issues: {n_issues}')
        print(f'  Clean: {len(results["items"]) - n_issues}')

        # Issue breakdown
        issue_types = defaultdict(int)
        for it in results['items']:
            for iss in it['issues']:
                # categorize
                if 'too small' in iss:
                    issue_types['bbox too small'] += 1
                elif 'too large' in iss:
                    issue_types['bbox too large'] += 1
                elif 'non-convex' in iss or 'crossed' in iss:
                    issue_types['non-convex quad'] += 1
                elif 'out of [0,1]' in iss:
                    issue_types['out of range'] += 1
                elif 'expected' in iss and 'fields' in iss:
                    issue_types['wrong field count'] += 1
                elif 'Empty label' in iss:
                    issue_types['empty label'] += 1
                elif 'Duplicate' in iss:
                    issue_types['duplicate image'] += 1
                elif 'Image' in iss:
                    issue_types['image error'] += 1
                else:
                    issue_types['other'] += 1

        if issue_types:
            print('\n  Issue breakdown:')
            for k, v in sorted(issue_types.items(), key=lambda x: -x[1]):
                print(f'    {k}: {v}')

        html_path = os.path.join(os.path.dirname(args.zip) or '.', 'dataset_review.html')
        generate_review_html(results, work_dir, html_path)

        print(f'\n✅ Review complete!')
        print(f'   Open {html_path} in browser to review')
        print(f'   Mark items → Export delete list → Re-run with --delete')

        # Don't clean up work dir yet (HTML needs image paths)
        print(f'   (Work dir: {work_dir} — delete after review)')


if __name__ == '__main__':
    main()
