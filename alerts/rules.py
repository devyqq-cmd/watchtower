from dataclasses import dataclass, field
from typing import Dict

@dataclass(frozen=True)
class AlertConfig:
    # --- Classic Alerts (Event-based) ---
    vol_percentile_hi: float = 0.90
    volume_z_hi: float = 3.0

    # Structural break: break recent high/low with volume confirmation
    breakout_lookback: int = 30
    breakout_volume_z: float = 2.2

    # Tail risk: large daily move with rising vol (simple proxy)
    tail_ret_hi: float = 0.06     # +6% daily
    tail_ret_lo: float = -0.06    # -6% daily

    # --- Risk Scoring (Condition-based) ---
    # 0 to 100 Score Components
    ema_fast: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    
    # Overextension: Price distance from EMA200
    z_score_lookback: int = 100
    z_threshold_greed: float = 2.0  # Sell signal when price is too far above average
    z_threshold_fear: float = -2.0 # Buy signal when price is too far below average

    # --- Global Market State (VIX) ---
    vix_ticker: str = "^VIX"
    vix_panic_threshold: float = 30.0
    vix_complacency_threshold: float = 13.0

    # --- Fundamental Valuation (Value Anchors) ---
    pe_high_threshold: float = 40.0   # Overvalued for typical large caps
    pe_low_threshold: float = 12.0    # Undervalued area
    pb_high_threshold: float = 5.0
    div_yield_min: float = 0.03       # 3% yield is a strong support for long-termers

    # Weights for the Composite Risk Score
    weights: Dict[str, float] = field(default_factory=lambda: {
        "trend": 0.25,
        "overextension": 0.3,
        "momentum": 0.15,
        "volatility": 0.1,
        "valuation": 0.2
    })

    @classmethod
    def load_evolution(cls) -> "AlertConfig":
        """
        Loads the 'Evolved' configuration from past learnings.
        """
        import json
        import os
        base = cls()
        path = "data/evolution.json"
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    evo = json.load(f)
                    # Override weights if present
                    if "weights" in evo:
                        base.weights.update(evo["weights"])
                    # Override thresholds if present
                    for k, v in evo.get("thresholds", {}).items():
                        if hasattr(base, k):
                            setattr(base, k, v)
                    print(f"[System] 🧬 Loaded Evolved Parameters from {path}")
            except Exception as e:
                print(f"[System] Failed to load evolution: {e}")
        return base

    # --- Engine Controls ---
    # de-dupe
    cooldown_hours: int = 6

    # minimum history required to avoid noisy alerts (new listings)
    # Lowered to 100 for better initial responsiveness
    min_bars: int = 100
