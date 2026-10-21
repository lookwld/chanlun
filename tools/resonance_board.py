"""共振扫描看板板块 —— 生成**独立预览页**, 并持有可直接粘进实盘看板的片段。

用户口径(2026-09-29): 「把扫描的结果在 web 看板中, 单独开一个版块列出来」。

实盘看板是 `chan-quant/app/web.py`(单文件 http.server + 内嵌 `_PAGE` 字符串 +
`collect_data()`)。要加一个板块只需三处改动, 本文件把这三处**原样实现一遍**
并渲染成独立 HTML, 好处是:

* 不必碰服务器就能先看到效果(`out/resonance_board.html`);
* 真要上线时, 把下面三个常量原样搬进 `web.py` 即可, 不靠人重写。

三处改动(见 `REPORT §部署包`):
  1. `_PAGE` 的 `<main>` 里加一行锚点   → `ANCHOR`
  2. `<script>` 里加渲染函数            → `RESO_JS`
  3. `collect_data()` 里加 4 行读 JSON  → `collect_snippet()`

⚠ 本工具只**读** `out/resonance_scan.json`, 不联网、不下单、不写任何实盘目录。

运行::

  python tools/resonance_board.py --json out/resonance_scan.json \\
      --out out/resonance_board.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------------------------------------------ 拼装块
# 1) 锚点: 加进 web.py `_PAGE` 的 <main> 里(放在"交易所实际持仓"之前)
ANCHOR = """<h2>共振扫描 · 4h+1h 长短期共振 (只做多)<span class="badge" id="reso_ts"></span></h2>
<div id="reso"></div>
"""

# 2) 渲染函数: 加进 web.py `_PAGE` 的 <script> 里, 由 load() 调用 resoHTML(d.resonance)
RESO_JS = r"""
function resoRow(h){
 var lp=h.s_price||h.l_price||0;
 var dd=lp?((h.entry/lp-1)*100).toFixed(2)+'%':'—';
 return [esc(h.sym),esc(h.label),(h.entry==null?'—':h.entry),dd,
   esc(h.l_trend||'—')+'/'+esc(h.s_trend||'—'),(h.age_h==null?'—':h.age_h.toFixed(1)+'h')];
}
function resoHTML(d){
 if(!d)return '<div class="empty">扫描结果缺失 (尚未运行 tools/scan_resonance.py)</div>';
 var s=d.signals||[],p=d.pending||[],r=d.recent||[];
 var c=function(k,v,sub){return '<div class="card"><div class="k">'+k+'</div><div class="v">'+v+
   '</div>'+(sub?'<div class="sub">'+sub+'</div>':'')+'</div>'};
 var cards=c('当前共振信号',s.length+' 条',
     '最近 '+d.fresh_bars+' 根 '+esc(d.s_tf)+' 内')+
   c('观察名单',p.length+' 个','长周期买点已成立, 等短周期给点')+
   c('近期信号',r.length+' 条','最近 '+(d.recent_days||30)+' 天')+
   c('池子',(d.n_alive||0)+' / '+(d.n_symbols||0)+' 个','仍在更新的归档标的');
 var warn=(d.now_ref_str?'<div class="empty">数据时点 '+esc(d.now_ref_str)+
   ' · 生成于 '+esc(d.generated_at)+' · 规则: 两级同时出买点且至少一侧买①</div>':'');
 var t1=s.length?tbl(['品种','共振组合','入场价','距结构低点','长/短趋势','信号年龄'],s.map(resoRow))
   :'<div class="empty">当前无共振信号 —— 该口径信号天然稀疏(全市场约 15 条/年)</div>';
 var t2=p.length?tbl(['品种','长周期买点','现价','结构低点','距低点','已过'],
   p.map(function(x){var l=x.l_price||0;return [esc(x.sym),esc(x.l_kind),
     (x.last_close==null?'—':x.last_close),(l?((x.last_close/l-1)*100).toFixed(2)+'%':'—'),
     (x.age_h==null?'—':x.age_h.toFixed(1)+'h')]}))
   :'<div class="empty">观察名单为空</div>';
 var t3=r.length?tbl(['品种','共振组合','触发时间','入场价','距结构低点','年龄'],
   r.map(function(h){var o=resoRow(h);return [o[0],o[1],
     new Date(h.ts).toLocaleString(),o[2],o[3],(h.age_h/24).toFixed(1)+'d']}))
   :'<div class="empty">最近 '+(d.recent_days||30)+' 天无共振信号</div>';
 return warn+'<div class="cards">'+cards+'</div>'+
  '<h3>当前共振信号</h3>'+t1+
  '<h3>观察名单 (长周期已出买点, 等短周期触发)</h3>'+t2+
  '<h3>近期信号 (最近 '+(d.recent_days||30)+' 天)</h3>'+t3;
}
"""

# 3) collect_data() 里加这几行(在 return 之前), 并把 "resonance": _reso 放进返回值
COLLECT_SNIPPET = '''    # ---- 共振扫描板块: 读扫描程序落盘的 JSON(只读, 缺失返回 None) ----
    _reso = None
    try:
        _p = Path(cfg.get("web", {}).get("resonance_json",
                                        "data/resonance_scan.json"))
        if _p.exists():
            _reso = json.loads(_p.read_text(encoding="utf-8"))
    except Exception:
        _reso = None
    ...
    return {..., "resonance": _reso}
'''

_DASH_CSS = """
body{font-family:-apple-system,'Segoe UI',Roboto,'Microsoft YaHei',sans-serif;
margin:0;background:#0f1420;color:#dde3ee}
header{padding:14px 20px;background:#161d2e;border-bottom:1px solid #232c42}
h1{font-size:17px;margin:0}
.badge{font-size:12px;padding:3px 10px;border-radius:20px;background:#232c42;margin-left:8px}
main{padding:16px;max-width:1100px;margin:0 auto}
h2{font-size:14px;margin:22px 0 8px;color:#8fa3c8;text-transform:uppercase;letter-spacing:1px}
h3{font-size:15px;margin:26px 0 4px;padding-left:10px;border-left:3px solid #3b82f6}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.card{background:#161d2e;border:1px solid #232c42;border-radius:10px;padding:12px}
.card .k{font-size:11px;color:#8fa3c8}.card .v{font-size:19px;margin-top:4px;font-weight:600}
.card .sub{font-size:10.5px;color:#8fa3c8;font-weight:400;margin-top:3px;line-height:1.35}
table{width:100%;border-collapse:collapse;font-size:12.5px;background:#161d2e;
border-radius:10px;overflow:hidden;border:1px solid #232c42}
th,td{padding:7px 9px;text-align:left;border-bottom:1px solid #232c42;white-space:nowrap}
th{color:#8fa3c8;font-weight:500;font-size:11.5px}
.empty{color:#5b6b8c;padding:14px;text-align:center}
.note{color:#8fa3c8;font-size:11.5px;padding:8px 2px;line-height:1.5}
"""


def build_preview(doc: dict | None) -> str:
    data = json.dumps(doc, ensure_ascii=False) if doc else "null"
    ts = (doc or {}).get("generated_at", "—")
    return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>共振扫描 · 看板板块预览</title>
<style>{_DASH_CSS}</style></head><body>
<header><h1>📊 看板 (本地预览 —— 仅共振扫描板块)</h1></header>
<main>
<div class="note">这是要加进实盘看板的新板块的**本地预览**。
样式与 <code>chan-quant/app/web.py</code> 完全一致; 数据来自
<code>out/resonance_scan.json</code>(生成于 {ts})。此页为静态快照, 不联网、不下单。</div>
{ANCHOR}
</main>
<script>
const _D = {data};
function esc(s){{return String(s==null?'':s).replace(/[&<>]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]))}}
function tbl(h,rows){{if(!rows.length)return '<div class="empty">暂无数据</div>';
return '<table><tr>'+h.map(x=>'<th>'+x+'</th>').join('')+'</tr>'+
rows.map(function(r){{var c=r.slice(0,h.length);while(c.length<h.length)c.push('—');
return '<tr>'+c.map(x=>'<td>'+x+'</td>').join('')+'</tr>'}}).join('')+'</table>'}}
{RESO_JS}
document.getElementById('reso').innerHTML=resoHTML(_D);
document.getElementById('reso_ts').textContent=_D&&_D.generated_at?('生成于 '+_D.generated_at):'';
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="out/resonance_scan.json")
    ap.add_argument("--out", default="out/resonance_board.html")
    ap.add_argument("--emit-patch", default="",
                    help="把 web.py 的三处改动写到这个文件(默认不写)")
    args = ap.parse_args()

    doc = None
    if os.path.exists(args.json):
        try:
            with open(args.json, encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception as e:                                    # noqa: BLE001
            print(f"读 {args.json} 失败: {e}")
    else:
        print(f"⚠ 找不到 {args.json}; 将渲染'缺失'态。先跑 tools/scan_resonance.py")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(build_preview(doc))
    print(f"预览页 → {args.out}")
    if doc:
        print(f"  当前信号 {doc.get('n_signals')} 条 · 观察名单 "
              f"{len(doc.get('pending') or [])} 个 · 近期 "
              f"{doc.get('n_recent')} 条 · 数据时点 {doc.get('now_ref_str')}")

    if args.emit_patch:
        with open(args.emit_patch, "w", encoding="utf-8") as fh:
            fh.write("# web.py 三处改动(原样粘贴)\n\n")
            fh.write("## 1) _PAGE 的 <main> 里加锚点\n```html\n" + ANCHOR + "```\n\n")
            fh.write("## 2) <script> 里加渲染函数\n```js\n" + RESO_JS + "\n```\n\n")
            fh.write("## 3) collect_data() 里读 JSON\n```python\n"
                     + COLLECT_SNIPPET + "```\n")
        print(f"web.py 补丁 → {args.emit_patch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
