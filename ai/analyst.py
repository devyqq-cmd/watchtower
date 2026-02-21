from __future__ import annotations
import os
from typing import Dict, Any

import httpx

PROMPT_TEMPLATE = """你是专业的港美股风险管理顾问。根据以下量化指标，用中文给出200字以内的深度分析。

标的：{symbol}
触发规则：{rule_id}（{severity}级别）
风险评分：{risk_score:.0f}/100
RSI：{rsi:.1f}
价格偏离EMA200的Z-score：{z_dist:.2f}
当前价格：{price:.2f}

要求：
1. 先解释触发原因（量化指标说明什么）
2. 结合历史规律给出风险判断
3. 按持仓时间（短/长期）分别给出具体操作建议
输出格式：直接给正文，不加标题，不超过200字。"""

_MINIMAX_URL = "https://api.minimax.chat/v1/text/chatcompletion_v2"
_MINIMAX_MODEL = "MiniMax-Text-01"


class AINarrativeAnalyst:
    """
    Translates quantitative risks into human-readable narratives.
    Priority: MiniMax API → Anthropic API → rule-based fallback.
    """

    def __init__(self, api_key: str = None):
        # api_key param kept for backward compat but not used
        self.minimax_key = os.getenv("MINIMAX_API_KEY", "").strip()
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    def analyze_risk_context(self, symbol: str, risk_data: Dict[str, Any], news_context: str = "") -> str:
        score = risk_data.get("risk_score", 0)
        rsi = risk_data.get("rsi", 50)
        z = risk_data.get("z_dist", 0)
        price = risk_data.get("price", 0)
        rule_id = risk_data.get("rule_id", "UNKNOWN")
        severity = risk_data.get("severity", "med")

        prompt = PROMPT_TEMPLATE.format(
            symbol=symbol,
            rule_id=rule_id,
            severity=severity,
            risk_score=score,
            rsi=rsi,
            z_dist=z,
            price=price,
        )

        if self.minimax_key:
            return self._call_minimax(prompt, symbol, score, rsi, z)

        if self.anthropic_key:
            return self._call_anthropic(prompt, symbol, score, rsi, z)

        return self._fallback_rule_engine(symbol, score, rsi, z)

    def _call_minimax(self, prompt: str, symbol: str, score: float, rsi: float, z: float) -> str:
        try:
            resp = httpx.post(
                _MINIMAX_URL,
                headers={
                    "Authorization": f"Bearer {self.minimax_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": _MINIMAX_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 400,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[analyst] MiniMax API error: {e}, falling back to rule engine")
            return self._fallback_rule_engine(symbol, score, rsi, z)

    def _call_anthropic(self, prompt: str, symbol: str, score: float, rsi: float, z: float) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.anthropic_key)
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            print(f"[analyst] Anthropic API error: {e}, falling back to rule engine")
            return self._fallback_rule_engine(symbol, score, rsi, z)

    def _fallback_rule_engine(self, symbol: str, score: float, rsi: float, z: float) -> str:
        if score > 70 and rsi > 75:
            return (
                f"🚨 AI ADVICE: {symbol} is currently in a 'GREED TRAP'. "
                "Quant factors are overextended. This is a classic zone where human greed "
                "leads to round-tripping profits. ACTION: HARVEST PROFITS."
            )
        if score > 60 and rsi < 20 and z < -2.0:
            return (
                f"🛡️ AI ADVICE: {symbol} is experiencing EXTREME PANIC (RSI {rsi:.1f}). "
                "While quant risk is high due to price velocity, this often marks a "
                "'Selling Exhaustion' point for long-term holders. ACTION: DO NOT PANIC SELL. OBSERVE REVERSAL."
            )
        return "✅ AI ADVICE: Market noise within normal parameters. Maintain long-term position and ignore short-term fluctuations."
