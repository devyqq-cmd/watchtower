from __future__ import annotations
import os
import json
from typing import Dict, Any

class AINarrativeAnalyst:
    """
    The 'AI Brain' that translates quantitative risks into human-readable 
    narratives and verifies the fundamental signal.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def analyze_risk_context(self, symbol: str, risk_data: Dict[str, Any], news_context: str = "") -> str:
        """
        Synthesizes Quant + Narrative.
        Args:
            symbol: Ticker (e.g. 0700.HK)
            risk_data: {score, rsi, z_dist, price}
            news_context: A summary of recent news (fetched via tools/search)
        """
        score = risk_data.get("risk_score", 0)
        rsi = risk_data.get("rsi", 50)
        z = risk_data.get("z_dist", 0)
        
        # Simple Logic for Local-only Mode (If API not available)
        if not self.api_key:
            return self._fallback_rule_engine(symbol, score, rsi, z)

        # In a real Industry-Standard app, we'd send this to an LLM:
        # prompt = f"Analyze {symbol} with risk_score {score} and RSI {rsi}. News: {news_context}"
        return "AI Analysis Pending (Waiting for LLM Prompt Integration)"

    def _fallback_rule_engine(self, symbol: str, score: float, rsi: float, z: float) -> str:
        """
        A rule-based AI reasoning engine for basic discipline enforcement.
        """
        if score > 70 and rsi > 75:
            return f"🚨 AI ADVICE: {symbol} is currently in a 'GREED TRAP'. Quant factors are overextended. This is a classic zone where human greed leads to round-tripping profits. ACTION: HARVEST PROFITS."
        
        if score > 60 and rsi < 20 and z < -2.0:
            return f"🛡️ AI ADVICE: {symbol} is experiencing EXTREME PANIC (RSI {rsi:.1f}). While quant risk is high due to price velocity, this often marks a 'Selling Exhaustion' point for long-term holders. ACTION: DO NOT PANIC SELL. OBSERVE REVERSAL."
            
        return "✅ AI ADVICE: Market noise within normal parameters. Maintain long-term position and ignore short-term fluctuations."

