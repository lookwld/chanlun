"""DeepSeek 大模型筛选 —— 复刻 `chan-quant/app/llm.py` 的契约。

对候选信号做一次 LLM 二次筛选(可选, 默认关闭)。dry_run / 未配置 key / 未启用时
一律放行(不拦截), 与 chan-quant "默认关闭、回测先行" 的纪律一致。

⚠ 需要 `openai` SDK 或手动 http 请求; 未安装/未配置时静默降级为放行。
"""

from __future__ import annotations

import json
import os
from urllib import request as _ur

__all__ = ["LLMFilter"]


class LLMFilter:
    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.cfg = cfg.get("llm", {}) or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.key = os.getenv(self.cfg.get("api_key_env", "DEEPSEEK_API_KEY"), "")
        self.base_url = self.cfg.get("base_url", "https://api.deepseek.com")
        self.model = self.cfg.get("model", "deepseek-chat")

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.key)

    def filter(self, sig: dict) -> tuple[bool, str]:
        """返回 (放行, 理由)。未启用/未配置 -> (True, "llm_disabled")。"""
        if not self.active:
            return True, "llm_disabled"
        prompt = (
            "你是量化风控助手。基于以下缠论信号判断是否值得做多/做空, "
            "只回答 YES 或 NO。\n"
            f"信号: {json.dumps(sig, ensure_ascii=False)}"
        )
        try:
            body = json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 4,
            }).encode("utf-8")
            req = _ur.Request(
                self.base_url.rstrip("/") + "/chat/completions",
                data=body,
                headers={"Authorization": f"Bearer {self.key}",
                         "Content-Type": "application/json"},
            )
            with _ur.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"].strip().upper()
            ok = text.startswith("YES")
            return ok, ("llm_ok" if ok else "llm_reject")
        except Exception as e:  # 网络/格式异常 -> fail-open(放行)
            return True, f"llm_error:{type(e).__name__}"
