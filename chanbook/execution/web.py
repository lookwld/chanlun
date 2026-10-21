"""HTTP 看板 —— 复刻 `chan-quant/app/web.py` 的接口契约。

提供: 账户快照 / 持仓 / 近期成交 / 复盘报告 / 日志尾巴 / 手动平仓 / 清空记录。
数据来源: 传入的 SpotTrader / FuturesTrader 实例的 snapshot() 与 Analyzer。

⚠ 单文件标准库实现(无 Flask), 与 chan-quant 一致(BaseHTTPRequestHandler)。
   默认绑定 127.0.0.1, 不对外暴露。
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

__all__ = ["serve", "collect_data"]


def _human(n: float) -> str:
    return f"{n:+,.4f}"


def collect_data(spot=None, fut=None) -> dict:
    """聚合看板数据(spot/fut 为 SpotTrader/FuturesTrader 实例或 None)。"""
    out = {"spot": None, "futures": None}
    if spot is not None:
        snap = spot.snapshot()
        out["spot"] = {
            "account": snap["account"],
            "open_count": snap["open_count"],
            "positions": snap["positions"],
        }
    if fut is not None:
        snap = fut.snapshot()
        out["futures"] = {
            "account": snap["account"],
            "leverage": snap.get("leverage"),
            "open_count": snap["open_count"],
            "positions": snap["positions"],
        }
    return out


def _render(data: dict) -> str:
    rows = []
    rows.append("<html><head><meta charset='utf-8'><title>chanbook 看板</title>")
    rows.append("<style>body{font-family:monospace;margin:24px;background:#0f1115;"
                "color:#e6e6e6} h2{color:#7dd3fc} table{border-collapse:collapse;"
                "margin:8px 0} td,th{border:1px solid #333;padding:4px 10px;"
                "font-size:13px} th{background:#1f2937}</style></head><body>")
    rows.append("<h2>chanbook 执行看板</h2>")

    for land, key in (("现货", "spot"), ("合约", "futures")):
        d = data.get(key)
        if not d:
            continue
        acc = d.get("account", {})
        rows.append(f"<h2>{land}</h2>")
        rows.append("<table>")
        for k, v in acc.items():
            rows.append(f"<tr><td>{k}</td><td>{_human(v) if isinstance(v, float) else v}</td></tr>")
        rows.append(f"<tr><td>open_count</td><td>{d.get('open_count')}</td></tr>")
        if land == "合约" and d.get("leverage"):
            rows.append(f"<tr><td>leverage</td><td>{d.get('leverage')}</td></tr>")
        rows.append("</table>")
        pos = d.get("positions", [])
        if pos:
            rows.append("<table><tr><th>symbol</th><th>side</th><th>qty</th>"
                        "<th>entry</th><th>stop</th><th>grade</th></tr>")
            for p in pos:
                rows.append(f"<tr><td>{p.get('symbol')}</td><td>{p.get('side','long')}</td>"
                            f"<td>{p.get('qty')}</td><td>{_human(p.get('entry',0))}</td>"
                            f"<td>{_human(p.get('stop',0))}</td><td>{p.get('grade','')}</td></tr>")
            rows.append("</table>")
    rows.append("</body></html>")
    return "\n".join(rows)


class _Handler(BaseHTTPRequestHandler):
    server_version = "chanbook/0.3"

    def _state(self):
        return getattr(self.server, "state", None) or {}

    def do_GET(self):
        state = self._state()
        if self.path in ("/", "/index.html"):
            body = _render(state.get("data", {})).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/data":
            body = json.dumps(state.get("data", {}), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/report":
            rep = ""
            for tr in state.get("traders", []):
                if tr is not None:
                    rep += tr.analyzer.report() + "\n\n"
            body = rep.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):  # 静默访问日志
        pass


def serve(cfg: dict, spot=None, fut=None, port: int = 8000) -> None:
    """启动看板(阻塞)。state 通过 setter 刷新; 简化版每次请求实时 collect。"""
    spot, fut = spot, fut

    def refresh():
        return collect_data(spot, fut)

    handler = _Handler
    httpd = HTTPServer(("127.0.0.1", port), handler)
    httpd.state = {"data": refresh(), "traders": [spot, fut]}

    class _RefreshHandler(_Handler):
        def _state(self):
            self.server.state = {"data": refresh(), "traders": [spot, fut]}
            return self.server.state

    httpd = HTTPServer(("127.0.0.1", port), _RefreshHandler)
    httpd.state = {"data": refresh(), "traders": [spot, fut]}
    print(f"chanbook 看板: http://127.0.0.1:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()
