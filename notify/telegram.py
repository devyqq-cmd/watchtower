from __future__ import annotations

import os
from typing import Dict, Any

import httpx

_SEVERITY_EMOJI = {
    "high": "🔴",
    "med": "🟡",
    "buy": "🟢",
}

_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


def send_alert(alert: Dict[str, Any]) -> bool:
    """
    发送单条告警到 Telegram。
    读取环境变量 TELEGRAM_BOT_TOKEN 和 TELEGRAM_CHAT_ID。
    任意一个缺失则静默跳过，不影响主流程。
    返回 True 表示发送成功，False 表示跳过或失败。
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        return False

    severity = alert.get("severity", "med")
    symbol = alert.get("symbol", "")
    rule_id = alert.get("rule_id", "")
    ctx = alert.get("context", {})
    risk_score = ctx.get("risk_score", 0)

    # 从 msg 字段提取 AI 建议（格式：原始消息 + \nAI ADVICE: ...）
    full_msg = alert.get("msg", "")
    ai_advice = ""
    if "\nAI ADVICE: " in full_msg:
        ai_advice = full_msg.split("\nAI ADVICE: ", 1)[1].strip()
    elif full_msg:
        ai_advice = full_msg.strip()

    emoji = _SEVERITY_EMOJI.get(severity, "⚪")
    text = (
        f"{emoji} {symbol} · {rule_id}\n"
        f"风险分：{risk_score:.0f}/100\n"
        f"AI建议：{ai_advice}"
    )

    url = _API_BASE.format(token=token)
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(url, json={"chat_id": chat_id, "text": text})
            if resp.status_code == 200:
                return True
            print(f"[telegram] 发送失败 HTTP {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"[telegram] 发送异常: {e}")
        return False
