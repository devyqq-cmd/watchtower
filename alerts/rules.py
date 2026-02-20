from dataclasses import dataclass

@dataclass(frozen=True)
class AlertConfig:
    # Event Shock: vol high percentile + volume spike
    vol_percentile_hi: float = 0.90
    volume_z_hi: float = 3.0

    # Structural break: break recent high/low with volume confirmation
    breakout_lookback: int = 30
    breakout_volume_z: float = 2.2

    # Tail risk: large daily move with rising vol (simple proxy)
    tail_ret_hi: float = 0.06     # +6% daily
    tail_ret_lo: float = -0.06    # -6% daily

    # de-dupe
    cooldown_hours: int = 6

    # minimum history required to avoid noisy alerts (new listings)
    min_bars: int = 80
