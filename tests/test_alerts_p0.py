import json

import pandas as pd
import pytest

from alerts.engine import calculate_risk_score
from alerts.rules import AlertConfig
from jobs.review import _normalize_weights


def test_load_evolution_applies_thresholds_and_weights(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    evo_path = data_dir / "evolution.json"
    evo_path.write_text(
        json.dumps(
            {
                "weights": {"trend": 0.3, "momentum": 0.1, "unknown": 9},
                "thresholds": {"cooldown_hours": 12, "min_bars": 140, "not_a_field": 1},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    cfg = AlertConfig.load_evolution()
    assert cfg.weights["trend"] == 0.3
    assert cfg.weights["momentum"] == 0.1
    assert "unknown" not in cfg.weights
    assert cfg.cooldown_hours == 12
    assert cfg.min_bars == 140


def test_oversold_context_does_not_inflate_sell_risk_score():
    cfg = AlertConfig()
    last_oversold = pd.Series(
        {
            "close": 90.0,
            "ema_fast": 95.0,
            "ema_slow": 100.0,
            "z_dist": -2.4,
            "rsi": 24.0,
            "vol_z": 0.5,
        }
    )
    last_overheated = pd.Series(
        {
            "close": 130.0,
            "ema_fast": 120.0,
            "ema_slow": 100.0,
            "z_dist": 2.3,
            "rsi": 82.0,
            "vol_z": 2.5,
        }
    )

    score_oversold = calculate_risk_score(last_oversold, cfg, global_vix=18.0, valuation={"trailingPE": 20.0})
    score_overheated = calculate_risk_score(last_overheated, cfg, global_vix=10.0, valuation={"trailingPE": 45.0})

    assert score_oversold < 50
    assert score_overheated > score_oversold


def test_normalize_weights_sums_to_one():
    """进化后权重必须精确归一化为 1.0，比例关系保持不变。"""
    # 模拟进化后的漂移权重（valuation +0.1, momentum -0.05 → 总和 1.05）
    drifted = {
        "trend": 0.25,
        "overextension": 0.3,
        "momentum": 0.10,
        "volatility": 0.1,
        "valuation": 0.30,
    }
    assert abs(sum(drifted.values()) - 1.0) > 0.01, "前提：漂移权重总和不为 1.0"

    normalized = _normalize_weights(drifted)

    assert abs(sum(normalized.values()) - 1.0) < 1e-9
    # 比例关系不变：valuation 仍应大于 momentum
    assert normalized["valuation"] > normalized["momentum"]
    # 所有键保留
    assert set(normalized.keys()) == set(drifted.keys())


def test_normalize_weights_preserves_already_normalized():
    """权重已归一化时，归一化操作应为幂等的。"""
    weights = {"trend": 0.25, "overextension": 0.3, "momentum": 0.15, "volatility": 0.1, "valuation": 0.2}
    assert abs(sum(weights.values()) - 1.0) < 1e-9

    normalized = _normalize_weights(weights)

    for k in weights:
        assert abs(normalized[k] - weights[k]) < 1e-9
