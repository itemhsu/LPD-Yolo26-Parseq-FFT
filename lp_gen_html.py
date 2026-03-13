#!/usr/bin/env python3
"""
lp_gen_html.py — 從已分離圖片的 results.json 產生輕量 HTML 檢視器
=================================================================
用法：
    python lp_gen_html.py
    python lp_gen_html.py --json /path/to/results.json --output-dir /path/to/output/

圖片位於 output-dir/plate_images/ 下，檔名格式如 19676_1_orig.png
"""

import os, sys, json, re, argparse

RESUL_JSON = '/Users/miniaicar/amtk/lp/lp_viewer_tool/lp_viewer_output/results.json'
OUTPUT_DIR = '/Users/miniaicar/amtk/lp/lp_viewer_tool/lp_viewer_output/'
IMG_DIR    = 'plate_images'  # 相對於 HTML 的子目錄


def extract_plate_from_filename(path):
    """從 original_path 檔名中抽取車牌號碼"""
    fn = os.path.basename(path)
    fn_noext = os.path.splitext(fn)[0]

    # -event-PLATE.jpg
    m = re.search(r'-event-([A-Z0-9]+)', fn_noext, re.IGNORECASE)
    if m: return m.group(1).upper()

    # -Occupy-PLATE
    m = re.search(r'-[Oo]ccupy-([A-Z0-9]+)', fn_noext, re.IGNORECASE)
    if m: return m.group(1).upper()

    # _PLATE_1.jpg  (letters+digits, 4-8 chars)
    m = re.search(r'_([A-Z]{2,4}\d{2,5}[A-Z]*)(?:_\d+)?$', fn_noext, re.IGNORECASE)
    if m: return m.group(1).upper()

    # _DIGITS+LETTERS_ pattern like _299AU_
    m = re.search(r'_(\d{2,4}[A-Z]{1,3})(?:_\d+)?$', fn_noext, re.IGNORECASE)
    if m: return m.group(1).upper()

    # @full_PLATE_hash
    m = re.search(r'@full_([A-Z0-9]{4,8})_[a-f0-9]', fn_noext, re.IGNORECASE)
    if m:
        cand = m.group(1).upper()
        if re.search(r'[A-Z]', cand) and re.search(r'\d', cand):
            return cand

    # 0825E26@full_776GCE_...
    m = re.search(r'@full_(\d{3,4}[A-Z]{1,3})_', fn_noext, re.IGNORECASE)
    if m: return m.group(1).upper()

    # -NNN_PLATE_1
    m = re.search(r'-\d+-(\w+_)?([A-Z0-9]{4,8})(?:_\d+)?$', fn_noext, re.IGNORECASE)
    if m:
        cand = m.group(2).upper()
        if re.search(r'[A-Z]', cand) and re.search(r'\d', cand):
            return cand

    return ''


def generate_html(json_path, output_dir, per_page=100):
    print(f'Loading: {json_path}')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    items = [r for r in data if 'corrected_ocr' in r]
    items.sort(key=lambda r: abs(r.get('v_angle', 0)), reverse=True)
    total = len(items)
    print(f'  Total items: {total}')

    all_items = []
    for r in items:
        ocr = r.get('corrected_ocr', '')
        fn_plate = extract_plate_from_filename(r.get('original_path', ''))

        # 圖片路徑：已分離到 plate_images/  欄位是 _img_orig, _img_corr, _img_quad
        io = r.get('_img_orig', '')
        ic = r.get('_img_corr', '')
        iq = r.get('_img_quad', '')

        all_items.append({
            'p': r.get('plate_id', ''),
            'h': round(r.get('h_angle', 0), 6),
            'v': round(r.get('v_angle', 0), 6),
            'ch': round(r.get('corrected_h_angle', 0), 6),
            'cv': round(r.get('corrected_v_angle', 0), 6),
            'o': ocr,
            'cf': round(r.get('corrected_conf_first', 0), 6),
            'cl': round(r.get('corrected_conf_last', 0), 6),
            'io': io,
            'ic': ic,
            'iq': iq,
            'fn': fn_plate,
        })

    items_json = json.dumps(all_items, ensure_ascii=False, separators=(',', ':'))
    print(f'  Data payload: {len(items_json)/1024/1024:.1f} MB')

    html_path = os.path.join(output_dir, 'plate_viewer.html')

    html = r'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>車牌校正檢視器</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0c0c18;color:#ddd;padding:8px}

.hdr{text-align:center;padding:6px 0 2px}
.hdr h1{font-size:17px;color:#e94560;margin-bottom:2px;letter-spacing:1px}
.hdr .st{color:#777;font-size:11px}

/* ── Filter ── */
.fbar{
  background:linear-gradient(135deg,#13132a,#16213e);
  border:1px solid #252550;border-radius:8px;padding:10px 14px;
  margin:6px auto;max-width:1400px;
  display:flex;flex-wrap:wrap;gap:4px 16px;align-items:center;
}
.fbar .ft{color:#e94560;font-weight:bold;font-size:12px;margin-right:6px}
.fbar label{
  font-size:11px;color:#bbb;cursor:pointer;white-space:nowrap;
  display:flex;align-items:center;gap:3px;padding:3px 0;
  transition:color .15s;user-select:none;
}
.fbar label:hover{color:#fff}
.fbar input[type=checkbox]{accent-color:#e94560;width:14px;height:14px;cursor:pointer}

.fst{
  display:none;border-radius:5px;padding:6px 12px;margin:5px auto;max-width:1400px;
  font-size:11px;text-align:center;
  background:linear-gradient(90deg,#0a1628,#0f2040,#0a1628);
  border:1px solid #1a3060;color:#5dade2;
}
.fst.show{display:block}
.fst b{color:#e94560}

/* ── Pagination ── */
.pg{display:flex;justify-content:center;gap:3px;margin:6px 0;flex-wrap:wrap}
.pg button{
  background:#13132a;color:#999;border:1px solid #252550;padding:3px 9px;
  cursor:pointer;border-radius:3px;font-size:11px;min-width:28px;transition:all .12s;
}
.pg button.a{background:#e94560;color:#fff;border-color:#e94560;font-weight:bold}
.pg button:hover:not(:disabled){background:#1a3060;color:#fff;border-color:#3a5080}
.pg button:disabled{opacity:.25;cursor:default}

/* ── 10x10 Grid ── */
.g{display:grid;grid-template-columns:repeat(10,1fr);gap:3px;margin:4px 0}
@media(max-width:1100px){.g{grid-template-columns:repeat(7,1fr)}}
@media(max-width:700px){.g{grid-template-columns:repeat(4,1fr)}}

.c{
  background:#12122a;border:1px solid #1c1c38;border-radius:3px;
  cursor:pointer;overflow:hidden;display:flex;flex-direction:column;
  align-items:center;padding:2px 1px;transition:border-color .12s,box-shadow .12s;
}
.c:hover{border-color:#e94560;box-shadow:0 0 8px rgba(233,69,96,.25)}
.c .nm{
  font-size:9px;font-weight:700;color:#e94560;text-align:center;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;width:100%;
  padding:1px 2px;line-height:1.4;
}
.c .lb{font-size:7px;color:#3a3a55;line-height:1;margin:0}
.c img{width:100%;height:auto;display:block;min-height:10px;background:#0a0a16;transition:opacity .3s}
.c img.ld{opacity:.1}
.c .ck{position:absolute;top:2px;left:2px;z-index:3;accent-color:#2ecc71;width:14px;height:14px;cursor:pointer}
.c{position:relative}
.c.unchecked{opacity:.35}

/* ── Export bar ── */
.ebar{
  background:linear-gradient(135deg,#0f1e2a,#132a20);
  border:1px solid #1a4030;border-radius:6px;padding:8px 14px;
  margin:6px auto;max-width:1400px;
  display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;
}
.ebar span{color:#2ecc71;font-weight:bold;font-size:12px}
.ebar .cnt{color:#aaa;font-size:11px}
.ebar button{
  background:#1a3028;color:#ccc;border:1px solid #2a5040;padding:4px 12px;
  border-radius:4px;cursor:pointer;font-size:11px;transition:all .15s;
}
.ebar button:hover{background:#2ecc71;color:#000;border-color:#2ecc71}
.ebar button.exp{background:#e94560;border-color:#e94560;color:#fff;font-weight:bold}
.ebar button.exp:hover{background:#ff6b81}

/* ── Modal ── */
.ov{display:none;position:fixed;top:0;left:0;width:100%;height:100%;
  background:rgba(0,0,0,.92);z-index:1000;justify-content:center;align-items:center}
.ov.show{display:flex}
.ml{
  background:#13132e;border:1px solid #2a2a55;border-radius:10px;
  padding:18px 22px;max-width:900px;width:96%;max-height:93vh;
  overflow-y:auto;position:relative;
}
.ml .xx{position:absolute;top:6px;right:12px;font-size:24px;color:#e94560;
  cursor:pointer;line-height:1;z-index:2}
.ml .xx:hover{color:#ff6b81}
.ml h2{color:#e94560;font-size:17px;margin-bottom:10px}
.ml table{width:100%;border-collapse:collapse;margin-bottom:14px}
.ml th{color:#e94560;text-align:left;padding:4px 8px;font-size:11px;border-bottom:1px solid #1c1c40}
.ml td{padding:4px 8px;font-size:11px;border-bottom:1px solid #1c1c40;color:#bbb}
.ml .ok{color:#2ecc71} .ml .ng{color:#e94560}
.ml .cmp{display:flex;gap:14px;margin-bottom:14px;flex-wrap:wrap}
.ml .cmp .co{flex:1;min-width:180px}
.ml .cmp .co h3{color:#777;font-size:11px;margin-bottom:4px}
.ml .cmp .co img{width:100%;border-radius:4px;border:1px solid #2a2a55}
.ml .qs{margin-bottom:14px}
.ml .qs h3{color:#777;font-size:11px;margin-bottom:4px}
.ml .qs img{max-width:100%;border-radius:4px;border:1px solid #2a2a55}
.ml .nv{display:flex;justify-content:space-between;align-items:center;margin-top:12px}
.ml .nv button{background:#1a1a40;color:#ccc;border:1px solid #2a2a55;
  padding:5px 14px;border-radius:4px;cursor:pointer;font-size:12px;transition:all .15s}
.ml .nv button:hover:not(:disabled){background:#e94560;color:#fff}
.ml .nv button:disabled{opacity:.25;cursor:default}
.ml .nv span{color:#555;font-size:11px}
</style>
</head>
<body>

<div class="hdr">
  <h1>🚗 車牌校正結果檢視器</h1>
  <div class="st" id="st"></div>
</div>

<div class="fbar">
  <span class="ft">⚙ Filter:</span>
  <label><input type="checkbox" id="f1"> 濾除 v==0</label>
  <label><input type="checkbox" id="f2"> 濾除 OCR &lt; 5 字元</label>
  <label><input type="checkbox" id="f3"> 濾除 OCR ≠ 檔名</label>
  <label><input type="checkbox" id="f4"> 濾除 首/尾字信心 &lt; 0.99</label>
  <label><input type="checkbox" id="f5"> 濾除無數字</label>
  <label><input type="checkbox" id="f6"> 濾除含連續三個 7 或 9</label>
  <label><input type="checkbox" id="f7"> 濾除 |v| &lt; 0.0628</label>
  <label><input type="checkbox" id="f9"> 濾除 |v| == 0.0628</label>
  <label><input type="checkbox" id="f8"> 濾除含小寫/符號 (a-z . ,)</label>
</div>
<div class="fst" id="fst"></div>

<div class="ebar">
  <span>☑ 選取:</span>
  <span class="cnt" id="selCnt"></span>
  <button onclick="selAll()">全選本頁</button>
  <button onclick="deselAll()">取消本頁</button>
  <button onclick="selAllPages()">全選所有</button>
  <button onclick="deselAllPages()">取消所有</button>
  <button class="exp" onclick="exportJSON()">💾 輸出已勾選 JSON</button>
</div>

<div class="pg" id="p1"></div>
<div class="g" id="gd"></div>
<div class="pg" id="p2"></div>

<div class="ov" id="ov" onclick="if(event.target===this)hm()">
  <div class="ml" id="ml"></div>
</div>

<script>
const D=%%DATA%%;
const PP=%%PP%%;
const IM='%%IM%%/';
let F=[],cp=0,mi=-1;
/* 勾選狀態：key=plate_id, value=true/false。預設全選 */
const CK={};
function initCK(){for(let i=0;i<D.length;i++) if(!(D[i].p in CK)) CK[D[i].p]=true;}
initCK();
function updSelCnt(){
  const checked=F.filter(r=>CK[r.p]!==false).length;
  $('selCnt').textContent=checked+' / '+F.length+' 筆已勾選';
}

function $(id){return document.getElementById(id)}
function esc(s){return s?s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'):''}
function isr(s){return s?IM+s:''}

/* ═══ Filter ═══ */
function af(){
  const c1=$('f1').checked, c2=$('f2').checked, c3=$('f3').checked,
        c4=$('f4').checked, c5=$('f5').checked, c6=$('f6').checked,
        c7=$('f7').checked, c8=$('f8').checked, c9=$('f9').checked;
  let n1=0,n2=0,n3=0,n4=0,n5=0,n6=0,n7=0,n8=0,n9=0;
  F=[];
  for(let i=0;i<D.length;i++){
    const r=D[i];
    const ocr=r.o, fn=r.fn;
    if(c1 && r.v===0){n1++;continue}
    if(c2 && ocr.length<5){n2++;continue}
    if(c3 && fn && ocr.toUpperCase()!==fn){n3++;continue}
    if(c4 && (r.cf<0.99||r.cl<0.99)){n4++;continue}
    if(c5 && !/\d/.test(ocr)){n5++;continue}
    if(c6 && /[79]{3}/.test(ocr)){n6++;continue}
    if(c7 && Math.abs(r.v)<0.0628){n7++;continue}
    if(c9 && Math.abs(Math.abs(r.v)-0.0628)<0.00001){n9++;continue}
    if(c8 && /[a-z.,()\-+×÷=°%#@!?&*]/.test(ocr)){n8++;continue}
    F.push(r);
  }
  const tot=n1+n2+n3+n4+n5+n6+n7+n8+n9;
  const el=$('fst');
  if(tot>0){
    const ps=[];
    if(c1) ps.push('v==0: <b>-'+n1+'</b>');
    if(c2) ps.push('OCR<5字: <b>-'+n2+'</b>');
    if(c3) ps.push('OCR≠檔名: <b>-'+n3+'</b>');
    if(c4) ps.push('信心<0.99: <b>-'+n4+'</b>');
    if(c5) ps.push('無數字: <b>-'+n5+'</b>');
    if(c6) ps.push('連續7/9: <b>-'+n6+'</b>');
    if(c7) ps.push('|v|<0.0628: <b>-'+n7+'</b>');
    if(c9) ps.push('|v|==0.0628: <b>-'+n9+'</b>');
    if(c8) ps.push('含小寫/符號: <b>-'+n8+'</b>');
    el.innerHTML='已濾除 <b>'+tot+'</b> 筆（'+ps.join(' ｜ ')+'），剩餘 <b>'+F.length+'</b> 筆';
    el.classList.add('show');
  } else {
    el.classList.remove('show');
  }
  $('st').textContent=
    '全部 '+D.length+' 筆 ｜ 顯示 '+F.length+' 筆 ｜ 依 |v_angle| 降序 ｜ 每頁 '+PP+' ｜ 點擊查看詳情';
  cp=0;
  rp();
}

/* ═══ Pagination ═══ */
function tp(){return Math.max(1,Math.ceil(F.length/PP))}
function bph(){
  const t=tp();
  let h='<button onclick="gp('+(cp-1)+')"'+(cp<1?' disabled':'')+'>◀ 上頁</button>';
  const s=new Set();
  for(let i=0;i<Math.min(3,t);i++) s.add(i);
  for(let i=Math.max(0,t-3);i<t;i++) s.add(i);
  for(let i=Math.max(0,cp-2);i<=Math.min(t-1,cp+2);i++) s.add(i);
  let pv=-1;
  for(const p of[...s].sort((a,b)=>a-b)){
    if(pv>=0&&p-pv>1) h+='<button disabled>…</button>';
    h+='<button class="'+(p===cp?'a':'')+'" onclick="gp('+p+')">'+(p+1)+'</button>';
    pv=p;
  }
  h+='<button onclick="gp('+(cp+1)+')"'+(cp>=t-1?' disabled':'')+'>下頁 ▶</button>';
  return h;
}

/* ═══ Render Grid ═══ */
function rp(){
  const h=bph();
  $('p1').innerHTML=h;
  $('p2').innerHTML=h;
  const s=cp*PP, items=F.slice(s,s+PP), el=$('gd');

  el.innerHTML=items.map((r,i)=>{
    const gi=s+i;
    const chk=CK[r.p]!==false;
    return '<div class="c'+(chk?'':' unchecked')+'" data-ci="'+gi+'">'+
      '<input type="checkbox" class="ck" data-ck="'+gi+'"'+(chk?' checked':'')+
        ' onclick="tck(event,'+gi+')" title="勾選/取消">'+
      '<div class="nm" onclick="sm('+gi+')">'+esc(r.o||'—')+'</div>'+
      '<div class="lb">原始</div>'+
      '<img data-g="'+gi+'" data-f="io" class="ld" alt="" onclick="sm('+gi+')">'+
      '<div class="lb">校正</div>'+
      '<img data-g="'+gi+'" data-f="ic" class="ld" alt="" onclick="sm('+gi+')">'+
    '</div>';
  }).join('');

  /* 延遲載入：只載入本頁的圖片 */
  requestAnimationFrame(()=>{
    const imgs=el.querySelectorAll('img[data-g]');
    for(let k=0;k<imgs.length;k++){
      const img=imgs[k];
      const idx=parseInt(img.dataset.g);
      const fld=img.dataset.f;
      const r=F[idx];
      if(!r) continue;
      const url=isr(r[fld]);
      if(url){
        img.src=url;
        img.onload=function(){this.classList.remove('ld')};
        img.onerror=function(){this.style.background='#2a0a0a';this.classList.remove('ld')};
      } else {
        img.classList.remove('ld');
        img.style.background='#1a1a2a';
      }
    }
  });
  updSelCnt();
}

function gp(p){
  const t=tp();
  if(p<0||p>=t) return;
  cp=p;
  rp();
  window.scrollTo(0,0);
}

/* ═══ Checkbox 勾選 ═══ */
function tck(ev,gi){
  ev.stopPropagation();
  const r=F[gi];
  if(!r) return;
  const chk=ev.target.checked;
  CK[r.p]=chk;
  const card=ev.target.closest('.c');
  if(card){chk?card.classList.remove('unchecked'):card.classList.add('unchecked')}
  updSelCnt();
}
function selAll(){
  const s=cp*PP, items=F.slice(s,s+PP);
  items.forEach(r=>{CK[r.p]=true});
  rp();
}
function deselAll(){
  const s=cp*PP, items=F.slice(s,s+PP);
  items.forEach(r=>{CK[r.p]=false});
  rp();
}
function selAllPages(){
  F.forEach(r=>{CK[r.p]=true});
  rp();
}
function deselAllPages(){
  F.forEach(r=>{CK[r.p]=false});
  rp();
}
function exportJSON(){
  const checked=F.filter(r=>CK[r.p]!==false);
  if(checked.length===0){alert('沒有勾選任何車牌');return}
  const out=checked.map(r=>({
    plate_id:r.p,
    h_angle:r.h, v_angle:r.v,
    corrected_h_angle:r.ch, corrected_v_angle:r.cv,
    corrected_ocr:r.o,
    corrected_conf_first:r.cf, corrected_conf_last:r.cl,
    _img_orig:r.io, _img_corr:r.ic, _img_quad:r.iq,
    filename_plate:r.fn
  }));
  const blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');
  a.href=url;
  a.download='selected_plates_'+checked.length+'.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* ═══ Modal ═══ */
function sm(idx){
  mi=idx;
  rmm();
  $('ov').classList.add('show');
}

function rmm(){
  const r=F[mi];
  if(!r) return;
  const match=r.fn?(r.o.toUpperCase()===r.fn):true;
  const noFn=!r.fn;
  const ml=$('ml');
  ml.innerHTML=
    '<span class="xx" onclick="hm()">&times;</span>'+
    '<h2>#'+(mi+1)+' '+esc(r.o||'—')+'</h2>'+
    '<table>'+
      '<tr><th>項目</th><th>校正前</th><th>校正後</th></tr>'+
      '<tr><td>h 角度</td><td>'+r.h.toFixed(6)+'</td><td>'+r.ch.toFixed(6)+'</td></tr>'+
      '<tr><td>v 角度</td><td>'+r.v.toFixed(6)+'</td><td>'+r.cv.toFixed(6)+'</td></tr>'+
      '<tr><td>|v| 角度</td><td>'+Math.abs(r.v).toFixed(6)+'</td><td>'+Math.abs(r.cv).toFixed(6)+'</td></tr>'+
      '<tr><td>首字信心</td><td>—</td><td>'+r.cf.toFixed(6)+'</td></tr>'+
      '<tr><td>尾字信心</td><td>—</td><td>'+r.cl.toFixed(6)+'</td></tr>'+
      '<tr><td>OCR 字元數</td><td colspan="2">'+r.o.length+'</td></tr>'+
      '<tr><td>檔名車牌</td><td colspan="2">'+(r.fn||'<span style="color:#555">（無）</span>')+'</td></tr>'+
      '<tr><td>OCR ≡ 檔名</td><td colspan="2">'+
        (noFn?'<span style="color:#555">N/A</span>':
          (match?'<span class="ok">✓ 相符</span>':
                 '<span class="ng">✗ 不符 (OCR: '+esc(r.o)+' / 檔名: '+esc(r.fn)+')</span>'))+
      '</td></tr>'+
      '<tr><td>Plate ID</td><td colspan="2" style="font-size:10px;color:#444">'+esc(r.p)+'</td></tr>'+
    '</table>'+
    '<div class="cmp">'+
      '<div class="co"><h3>校正前投影</h3><img id="mio" alt=""></div>'+
      '<div class="co"><h3>校正後投影</h3><img id="mic" alt=""></div>'+
    '</div>'+
    '<div class="qs"><h3>原圖框定位置（紅＝原始  綠＝校正後）</h3><img id="miq" alt=""></div>'+
    '<div class="nv">'+
      '<button onclick="mn(-1)"'+(mi<=0?' disabled':'')+'> ◀ 上一筆</button>'+
      '<span>'+(mi+1)+' / '+F.length+'</span>'+
      '<button onclick="mn(1)"'+(mi>=F.length-1?' disabled':'')+'> 下一筆 ▶</button>'+
    '</div>';

  const u1=isr(r.io), u2=isr(r.ic), u3=isr(r.iq);
  if(u1) document.getElementById('mio').src=u1;
  if(u2) document.getElementById('mic').src=u2;
  if(u3) document.getElementById('miq').src=u3;
}

function mn(d){
  const ni=mi+d;
  if(ni<0||ni>=F.length) return;
  mi=ni;
  rmm();
  $('ml').scrollTop=0;
}

function hm(){$('ov').classList.remove('show');mi=-1}

/* ═══ Keyboard ═══ */
document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){hm();return}
  const inModal=$('ov').classList.contains('show');
  if(inModal){
    if(e.key==='ArrowLeft'){e.preventDefault();mn(-1)}
    if(e.key==='ArrowRight'){e.preventDefault();mn(1)}
  } else {
    if(e.key==='ArrowLeft') gp(cp-1);
    if(e.key==='ArrowRight') gp(cp+1);
  }
});

/* ═══ Init ═══ */
['f1','f2','f3','f4','f5','f6','f7','f8','f9'].forEach(id=>{
  $(id).addEventListener('change',af);
});
af();
</script>
</body>
</html>'''

    html = html.replace('%%DATA%%', items_json)
    html = html.replace('%%PP%%', str(per_page))
    html = html.replace('%%IM%%', IMG_DIR)

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    sz = os.path.getsize(html_path)
    print(f'\n  ✅ HTML saved: {html_path}')
    print(f'  📦 HTML size: {sz/1024/1024:.1f} MB')
    print(f'  🖼  Images from: {IMG_DIR}/')
    print(f'  📊 Total: {total} plates, {(total+per_page-1)//per_page} pages')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='產生車牌校正 HTML 檢視器')
    parser.add_argument('--json', type=str, default=RESUL_JSON)
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--per-page', type=int, default=100)
    args = parser.parse_args()
    generate_html(args.json, args.output_dir, args.per_page)
