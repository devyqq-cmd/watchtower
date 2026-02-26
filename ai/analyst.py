from __future__ import annotations
import os
import subprocess
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

DAILY_NARRATIVE_PROMPT = """你是专业港股分析师。根据以下数据生成一段 ≤200字 的中文市场点评（宏观+个股动态）：

VIX恐慌指数: {vix:.1f}（{vix_label}）
个股表现: {ticker_summary}
今日头条: {headlines}

要求：简洁、客观、点到为止，不要重复数字数据，突出最值得关注的风险或机会。"""

_MINIMAX_URL = "https://api.minimax.chat/v1/text/chatcompletion_v2"
_MINIMAX_MODEL = "MiniMax-Text-01"


class AINarrativeAnalyst:
    """
    Translates quantitative risks into human-readable narratives.
    Priority: Claude CLI (claude -p) → MiniMax API → Anthropic API → rule-based fallback.
    """

    def __init__(self, api_key: str = None):
        # api_key param kept for backward compat but not used
        self.minimax_key = os.getenv("MINIMAX_API_KEY", "").strip()
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self._claude_cli = self._detect_claude_cli()

    def _detect_claude_cli(self) -> str | None:
        """Return path to claude CLI if available, else None."""
        for candidate in ["claude", os.path.expanduser("~/.claude/local/claude")]:
            try:
                r = subprocess.run([candidate, "--version"], capture_output=True, timeout=5)
                if r.returncode == 0:
                    return candidate
            except Exception:
                pass
        return None

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

        if self._claude_cli:
            result = self._try_claude_cli(prompt)
            if result:
                return result

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

    def _try_claude_cli(self, prompt: str) -> str | None:
        """尝试调用 claude -p，成功返回结果，失败返回 None 让调用方降级。"""
        try:
            r = subprocess.run(
                [self._claude_cli, "-p", prompt],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
            print(f"[analyst] claude CLI failed (rc={r.returncode}), trying next engine")
            return None
        except Exception as e:
            print(f"[analyst] claude CLI exception: {e}, trying next engine")
            return None

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

    def generate_market_narrative(self, ctx: dict) -> str:
        """
        生成每日市场宏观叙事。
        ctx keys: vix, vix_label, tickers (list of dicts), alerts_count, news_headlines
        Priority: claude CLI → MiniMax → Anthropic → rule-based fallback.
        """
        vix = ctx.get("vix", 20.0)
        vix_label = ctx.get("vix_label", "中性")
        tickers = ctx.get("tickers", [])
        headlines = ctx.get("news_headlines", [])

        ticker_summary = "; ".join(
            f"{t['sym']} {t['pct_change']:+.1f}% 风险分{t['risk_score']:.0f}"
            for t in tickers
        )
        headlines_str = "、".join(headlines[:5]) if headlines else "暂无"

        prompt = DAILY_NARRATIVE_PROMPT.format(
            vix=vix,
            vix_label=vix_label,
            ticker_summary=ticker_summary,
            headlines=headlines_str,
        )

        if self._claude_cli:
            result = self._try_claude_cli(prompt)
            if result:
                return result

        if self.minimax_key:
            return self._call_minimax(prompt, "MARKET", vix, 0, 0)

        if self.anthropic_key:
            return self._call_anthropic(prompt, "MARKET", vix, 0, 0)

        return self._fallback_daily_narrative(ctx)

    def _fallback_daily_narrative(self, ctx: dict) -> str:
        """规则备选：无 AI 时生成简单中文市场点评。"""
        vix = ctx.get("vix", 20.0)
        vix_label = ctx.get("vix_label", "中性")
        tickers = ctx.get("tickers", [])
        alerts_count = ctx.get("alerts_count", 0)

        gainers = [t for t in tickers if t.get("pct_change", 0) > 0]
        losers = [t for t in tickers if t.get("pct_change", 0) < 0]

        if len(gainers) > len(losers):
            trend = "整体偏多，多数标的收涨"
        elif len(losers) > len(gainers):
            trend = "整体承压，多数标的收跌"
        else:
            trend = "涨跌互现，市场分歧明显"

        high_risk = [t for t in tickers if t.get("risk_score", 0) > 60]
        risk_note = ""
        if high_risk:
            syms = "、".join(t["sym"] for t in high_risk)
            risk_note = f"注意 {syms} 风险分偏高，请控制仓位。"

        alert_note = f"今日触发 {alerts_count} 条告警。" if alerts_count > 0 else ""

        return f"今日港股{trend}，VIX {vix:.1f}（{vix_label}）。{risk_note}{alert_note}".strip()
