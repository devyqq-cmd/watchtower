import sys
from pathlib import Path

# Add project root to sys.path for robust imports
root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import os
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import json
from pathlib import Path
from app.providers.store.sqlite_store import SQLiteStore
from jobs.backtest import run_backtest
from alerts import AlertConfig, compute_features, calculate_risk_score


def get_db_path() -> str:
    env_path = os.getenv("WATCHTOWER_DB_PATH")
    if env_path:
        return env_path
    return "watchtower.db"


def get_connection(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def load_tickers(conn: sqlite3.Connection) -> list[str]:
    query = "SELECT DISTINCT ticker FROM prices ORDER BY ticker"
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        return []
    return df["ticker"].tolist()


def load_price_data(
    conn: sqlite3.Connection,
    ticker: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    base_query = "SELECT ts, open, high, low, close, volume FROM prices WHERE ticker = ?"
    params: list = [ticker]

    if start is not None:
        base_query += " AND ts >= ?"
        params.append(start.isoformat())
    if end is not None:
        base_query += " AND ts <= ?"
        params.append(end.isoformat())

    base_query += " ORDER BY ts"
    df = pd.read_sql_query(base_query, conn, params=params, parse_dates=["ts"])
    return df


RULE_LABELS = {
    "RISK_EXTREME": "极端风险预警",
    "RISK_HIGH":    "高风险预警",
    "RISK_LOW":     "低风险·可布局",
    "OVERSOLD_OPP": "超卖·买入机会",
    "EVENT_SHOCK":  "异常波动预警",
}


CSS = """
<style>
/* ── 全局 ── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1200px; }

/* ── Tab ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #f0f0f0; }
.stTabs [data-baseweb="tab"] {
    font-size: 14px; font-weight: 500; padding: 8px 20px;
    border-radius: 6px 6px 0 0; color: #555;
}
.stTabs [aria-selected="true"] { background: #e6f4ff !important; color: #1677ff !important; }

/* ── 指标卡 ── */
.kpi-card {
    padding: 16px 18px; border-radius: 10px;
    border-left: 4px solid #ccc; background: #fafafa;
    min-height: 100px; box-sizing: border-box;
}
.kpi-label {
    font-size: 11px; font-weight: 600; color: #888;
    text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px;
}
.kpi-value { font-size: 30px; font-weight: 700; color: #1a1a2e; line-height: 1.1; }
.kpi-sub   { font-size: 11px; font-weight: 500; margin-top: 5px; }

/* ── 操作建议卡 ── */
.advice-card {
    border-radius: 12px; padding: 20px 24px;
    border: 1px solid; margin-bottom: 4px;
}
.advice-title { font-size: 18px; font-weight: 700; margin-bottom: 14px; }
.advice-grid  { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.advice-section-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.6px; margin-bottom: 8px;
}
.advice-list { font-size: 13px; color: #444; margin: 0; padding-left: 16px; line-height: 1.9; }
.advice-action { font-size: 13px; color: #444; line-height: 1.9; }

/* ── 区块标题 ── */
.section-title {
    font-size: 15px; font-weight: 600; color: #1a1a2e;
    margin: 24px 0 4px 0; padding-bottom: 6px;
    border-bottom: 2px solid #f0f0f0;
}
.section-sub { font-size: 12px; color: #999; margin: 2px 0 10px 0; }
</style>
"""


def _kpi(col, label: str, value: str, sub: str, border: str, bg: str) -> None:
    col.markdown(
        f'<div class="kpi-card" style="border-left-color:{border}; background:{bg};">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub" style="color:{border};">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _advice_box(ticker: str, risk_val: float, rsi_val: float, z_val: float, price: float) -> None:
    """自定义 HTML 操作建议卡片：双栏（原因 + 执行建议），按风险等级配色。"""

    if risk_val >= 80:
        bg, border, text_color = "#fff1f0", "#ff4d4f", "#a8071a"
        icon, action = "🔴", "建议减仓止盈"
        reasons = [
            f"综合风险评分 {risk_val:.0f}/100，已进入极端危险区（&gt;80）",
            f"市场情绪过热，超买超卖指数 {rsi_val:.1f}",
            f"价格严重高于长期均线 {z_val:+.2f}，上涨动力接近透支",
        ]
        action_html = (
            f"建议<b>分批减仓 20%～30%</b>，先锁定部分利润。<br>"
            f"减仓后继续观察，等风险分回落到 60 以下再重新评估是否持仓。"
        )
    elif risk_val >= 60:
        bg, border, text_color = "#fff7e6", "#fa8c16", "#ad4e00"
        icon, action = "🟠", "建议控制仓位，保持观望"
        reasons = [
            f"综合风险评分 {risk_val:.0f}/100，已进入偏高区间（60～80）",
            f"超买超卖指数 {rsi_val:.1f}，偏高",
            f"价格偏离长期均线 {z_val:+.2f}，继续追高风险较大",
        ]
        action_html = (
            f"当前持仓<b>无需慌张</b>，但不建议加仓。<br>"
            f"等风险分回落到 50 以下、超买超卖回到正常区间再考虑操作。"
        )
    elif risk_val <= 25 or (rsi_val < 35 and z_val < -1.5):
        bg, border, text_color = "#f6ffed", "#52c41a", "#135200"
        icon, action = "🟢", "可考虑分批买入"
        reasons = []
        if risk_val <= 25:
            reasons.append(f"综合风险评分 {risk_val:.0f}/100，处于安全区间（&lt;25）")
        if rsi_val < 35:
            reasons.append(f"超买超卖指数 {rsi_val:.1f}，低于35，市场恐慌性超卖")
        if z_val < -1.5:
            reasons.append(f"价格偏离长期均线 {z_val:+.2f}，处于历史低估区间")
        action_html = (
            f"建议<b>分 2～3 批</b>逐步建仓，不要一次性全仓。<br>"
            f"例如：当前买入 1/3，若继续下跌 5%+ 再加仓 1/3，以此类推。<br>"
            f"长线持仓，耐心等待回升。"
        )
    else:
        bg, border, text_color = "#e6f4ff", "#1677ff", "#003a8c"
        icon, action = "⚪", "持仓观望，无需操作"
        reasons = [
            f"综合风险评分 {risk_val:.0f}/100，处于中性区间（25～60）",
            f"超买超卖指数 {rsi_val:.1f}，未出现明显超买或超卖",
            f"价格偏离长期均线 {z_val:+.2f}，在正常波动范围内",
        ]
        action_html = (
            f"维持现有仓位，<b>等待更明确的信号</b>再行动。<br>"
            f"系统持续监控，有变化会第一时间提示。"
        )

    reasons_html = "".join(f"<li>{r}</li>" for r in reasons)
    st.markdown(
        f'<div class="advice-card" style="background:{bg}; border-color:{border};">'
        f'<div class="advice-title" style="color:{text_color};">{icon}&nbsp; {ticker} &mdash; {action}</div>'
        f'<div class="advice-grid">'
        f'  <div>'
        f'    <div class="advice-section-label" style="color:{border};">触发原因</div>'
        f'    <ul class="advice-list">{reasons_html}</ul>'
        f'  </div>'
        f'  <div>'
        f'    <div class="advice-section-label" style="color:{border};">执行建议</div>'
        f'    <div class="advice-action">{action_html}</div>'
        f'  </div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="行情监控", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    # 自定义标题区
    st.markdown(
        '<h1 style="font-size:26px; font-weight:700; color:#1a1a2e; margin-bottom:4px;">'
        '📈 港美股行情监控</h1>'
        '<p style="font-size:13px; color:#888; margin-top:0; margin-bottom:16px;">'
        '数据每日收盘后自动更新 · 信号基于多因子量化模型</p>',
        unsafe_allow_html=True,
    )

    db_path = get_db_path()
    store = SQLiteStore(db_path=db_path)
    store.init_db()
    conn = get_connection(db_path)

    tickers = load_tickers(conn)
    if not tickers:
        st.warning("暂无数据，请先运行数据采集。")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.selectbox("选择股票", options=tickers)
    default_end   = datetime.utcnow()
    default_start = default_end - timedelta(days=90)
    with col2:
        start_date = st.date_input("起始日期", value=default_start.date())
    with col3:
        end_date = st.date_input("截止日期", value=default_end.date())

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(end_date,   datetime.max.time())

    df = load_price_data(conn, ticker, start=start_dt, end=end_dt)
    if df.empty:
        st.info("所选时间范围内暂无数据，请拉大日期范围。")
        return

    # ── 提前计算指标（操作建议和图表共用）──────────────────────────
    df_full   = load_price_data(conn, ticker)
    cfg       = AlertConfig.load_evolution()
    rsi_val   = z_val = risk_val = None
    feat_full = None

    if len(df_full) >= cfg.min_bars:
        feat_full = compute_features(df_full, cfg)
        last      = feat_full.iloc[-1]
        rsi_val   = float(last["rsi"])
        z_val     = float(last["z_dist"])
        risk_val  = calculate_risk_score(last, cfg, global_vix=19.0)

    # ── 操作建议（tabs 之前，第一眼就看到）─────────────────────────
    st.divider()
    if risk_val is not None:
        _advice_box(ticker, risk_val, rsi_val, z_val, df["close"].iloc[-1])
    else:
        st.info(f"数据积累不足（需 {cfg.min_bars} 根K线），暂无操作建议。")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 行情详情", "🔁 历史回测", "📋 信号记录"])

    # ── Tab 1：行情监控 ────────────────────────────────────────────
    with tab1:
        # 指标卡（颜色随信号变化）
        st.markdown('<div class="section-title">关键指标</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)

        _kpi(m1, "当前价格", f"{df['close'].iloc[-1]:.2f}", "港元 / 美元", "#1677ff", "#e6f4ff")

        if risk_val is not None:
            if risk_val >= 80:
                r_border, r_bg, r_sub = "#ff4d4f", "#fff1f0", "极端风险区 (>80)"
            elif risk_val >= 60:
                r_border, r_bg, r_sub = "#fa8c16", "#fff7e6", "偏高，注意控制仓位"
            elif risk_val <= 25:
                r_border, r_bg, r_sub = "#52c41a", "#f6ffed", "安全区间，可布局"
            else:
                r_border, r_bg, r_sub = "#8c8c8c", "#fafafa", "正常区间"
            _kpi(m2, "综合风险评分", f"{risk_val:.0f}", f"{r_sub}（满分100）", r_border, r_bg)
        else:
            _kpi(m2, "综合风险评分", "—", "数据不足", "#ccc", "#fafafa")

        if rsi_val is not None:
            if rsi_val < 30:
                i_border, i_bg, i_sub = "#52c41a", "#f6ffed", "超卖区间，跌过头了"
            elif rsi_val > 70:
                i_border, i_bg, i_sub = "#ff4d4f", "#fff1f0", "超买区间，涨过头了"
            else:
                i_border, i_bg, i_sub = "#8c8c8c", "#fafafa", "正常区间（30～70）"
            _kpi(m3, "超买超卖指数", f"{rsi_val:.1f}", i_sub, i_border, i_bg)
        else:
            _kpi(m3, "超买超卖指数", "—", "数据不足", "#ccc", "#fafafa")

        if z_val is not None:
            if z_val < -1.5:
                z_border, z_bg, z_sub = "#52c41a", "#f6ffed", "价格偏低，处于低估区间"
            elif z_val > 1.5:
                z_border, z_bg, z_sub = "#ff4d4f", "#fff1f0", "价格偏高，处于高估区间"
            else:
                z_border, z_bg, z_sub = "#8c8c8c", "#fafafa", "价格处于正常区间"
            _kpi(m4, "价格偏离均线", f"{z_val:+.2f}", z_sub, z_border, z_bg)
        else:
            _kpi(m4, "价格偏离均线", "—", "数据不足", "#ccc", "#fafafa")

        st.markdown("<br>", unsafe_allow_html=True)

        # 价格走势
        st.markdown(
            '<div class="section-title">价格走势</div>'
            '<div class="section-sub">收盘价 · 50日均线（近期趋势）· 200日均线（长期趋势）</div>',
            unsafe_allow_html=True,
        )
        if feat_full is not None:
            df_merged = df.merge(
                feat_full[["ts", "ema_fast", "ema_slow"]].rename(
                    columns={"ema_fast": "50日均线", "ema_slow": "200日均线"}
                ),
                on="ts", how="left",
            )
            st.line_chart(
                df_merged.set_index("ts")[["close", "50日均线", "200日均线"]].rename(columns={"close": "收盘价"}),
                height=280,
            )
        else:
            st.line_chart(df.set_index("ts")[["close"]].rename(columns={"close": "收盘价"}), height=280)

        # 超买超卖走势
        st.markdown(
            '<div class="section-title">超买超卖指数走势</div>'
            '<div class="section-sub">低于 30 = 超卖（跌过头，可能是买入机会）｜高于 70 = 超买（涨过头，注意回调）</div>',
            unsafe_allow_html=True,
        )
        if feat_full is not None:
            df_rsi = df.merge(feat_full[["ts", "rsi"]], on="ts", how="left")
            st.line_chart(df_rsi.set_index("ts")[["rsi"]].rename(columns={"rsi": "超买超卖指数"}), height=180)
        else:
            st.info("数据积累不足，暂时无法计算超买超卖指数。")

        # 近期价格明细
        st.markdown('<div class="section-title">近期每日价格</div>', unsafe_allow_html=True)
        df_display = df.tail(60).copy()
        df_display["ts"] = df_display["ts"].dt.strftime("%Y-%m-%d")
        df_display = df_display.rename(columns={
            "ts": "日期", "open": "开盘价", "high": "最高价",
            "low": "最低价", "close": "收盘价", "volume": "成交量",
        }).set_index("日期")
        st.dataframe(df_display, use_container_width=True, height=280)

        # 区间统计
        st.markdown('<div class="section-title">区间统计</div>', unsafe_allow_html=True)
        col_a, col_b, col_c, col_d = st.columns(4)
        _kpi(col_a, "最新收盘价",  f"{df['close'].iloc[-1]:.2f}", "最后交易日", "#1677ff", "#e6f4ff")
        _kpi(col_b, "区间最高价",  f"{df['high'].max():.2f}",    "所选时间范围", "#52c41a", "#f6ffed")
        _kpi(col_c, "区间最低价",  f"{df['low'].min():.2f}",     "所选时间范围", "#ff4d4f", "#fff1f0")
        _kpi(col_d, "区间总成交量", f"{df['volume'].sum():,.0f}", "所选时间范围", "#8c8c8c", "#fafafa")

    # ── Tab 3：信号记录 ────────────────────────────────────────────
    with tab3:
        st.subheader("历史信号记录")
        st.caption(
            "系统每次发出买入或卖出建议后，会在 20 个交易日后自动验证结果。"
            "  ✅ 判断正确  ❌ 判断错误  ⏳ 结果待定"
        )

        alerts_path = Path("data/alerts.jsonl")
        if not alerts_path.exists():
            st.info("尚无历史信号，请先运行数据采集。")
        else:
            raw_alerts = []
            for line in alerts_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        raw_alerts.append(json.loads(line))
                    except Exception:
                        pass

            if not raw_alerts:
                st.info("暂无信号记录。")
            else:
                OUTCOME_DAYS = 20
                OUTCOME_THRESHOLD = 0.05

                rows = []
                for a in raw_alerts:
                    ts_str          = a.get("ts", "")
                    symbol          = a.get("symbol", "")
                    rule_id         = a.get("rule_id", "")
                    severity        = a.get("severity", "")
                    ctx             = a.get("context", {})
                    risk_score      = ctx.get("risk_score", 0)
                    rsi             = ctx.get("rsi", None)
                    z_dist          = ctx.get("z_dist", None)
                    price_at_alert  = ctx.get("price", None)

                    is_buy    = severity == "buy" or rule_id in ("RISK_LOW", "OVERSOLD_OPP")
                    direction = "🟢 买入建议" if is_buy else "🔴 卖出建议"
                    rule_zh   = RULE_LABELS.get(rule_id, rule_id)

                    # 超买超卖指数解读
                    if rsi is not None:
                        rsi_note = f"{rsi:.1f}（超卖）" if rsi < 30 else (f"{rsi:.1f}（超买）" if rsi > 70 else f"{rsi:.1f}（正常）")
                    else:
                        rsi_note = "—"

                    # 价格偏离解读
                    if z_dist is not None:
                        z_note = f"{z_dist:+.2f}（严重低估）" if z_dist < -2 else (
                                 f"{z_dist:+.2f}（严重高估）" if z_dist > 2 else f"{z_dist:+.2f}（正常）")
                    else:
                        z_note = "—"

                    # 20日后结果
                    outcome_label = "⏳ 结果待定"
                    if ts_str:
                        try:
                            alert_ts   = datetime.fromisoformat(ts_str).replace(tzinfo=None)
                            outcome_ts = alert_ts + timedelta(days=OUTCOME_DAYS)
                            if outcome_ts < datetime.utcnow():
                                df_out = load_price_data(
                                    conn, symbol,
                                    start=alert_ts - timedelta(days=1),
                                    end=outcome_ts + timedelta(days=3),
                                )
                                if not df_out.empty and price_at_alert:
                                    df_out["ts"] = pd.to_datetime(df_out["ts"]).dt.tz_localize(None)
                                    after = df_out[df_out["ts"] >= pd.Timestamp(outcome_ts)]
                                    if not after.empty:
                                        p_end  = float(after.iloc[0]["close"])
                                        ret_20d = (p_end - float(price_at_alert)) / float(price_at_alert)
                                        if is_buy:
                                            if ret_20d > OUTCOME_THRESHOLD:
                                                outcome_label = f"✅ 判断正确  之后涨了 +{ret_20d:.1%}"
                                            elif ret_20d < -OUTCOME_THRESHOLD:
                                                outcome_label = f"❌ 判断错误  之后跌了 {ret_20d:.1%}"
                                            else:
                                                outcome_label = f"➖ 变化不大  {ret_20d:+.1%}"
                                        else:
                                            if ret_20d < -OUTCOME_THRESHOLD:
                                                outcome_label = f"✅ 判断正确  之后跌了 {ret_20d:.1%}"
                                            elif ret_20d > OUTCOME_THRESHOLD:
                                                outcome_label = f"❌ 判断错误  之后涨了 +{ret_20d:.1%}"
                                            else:
                                                outcome_label = f"➖ 变化不大  {ret_20d:+.1%}"
                        except Exception:
                            pass

                    rows.append({
                        "发出日期":   ts_str[:10] if ts_str else "",
                        "股票":       symbol,
                        "建议方向":   direction,
                        "触发原因":   rule_zh,
                        "综合风险分": f"{risk_score:.0f}",
                        "超买超卖":   rsi_note,
                        "价格偏离":   z_note,
                        "信号时价格": f"{price_at_alert:.2f}" if price_at_alert else "—",
                        "20日后结果": outcome_label,
                    })

                df_audit = pd.DataFrame(rows[::-1])

                total   = len(df_audit)
                success = df_audit["20日后结果"].str.startswith("✅").sum()
                failure = df_audit["20日后结果"].str.startswith("❌").sum()
                pending = df_audit["20日后结果"].str.startswith("⏳").sum()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("历史信号总数", total)
                c2.metric("✅ 判断正确", success)
                c3.metric("❌ 判断错误", failure)
                c4.metric("⏳ 结果待定", pending)

                if (success + failure) > 0:
                    win_rate = success / (success + failure)
                    st.progress(win_rate, text=f"信号准确率（已验证信号）：{win_rate:.1%}")

                st.dataframe(df_audit, use_container_width=True)

    # ── Tab 2：历史回测 ────────────────────────────────────────────
    with tab2:
        st.subheader("历史回测：AI 策略 vs. 买入持有")
        st.caption("模拟：如果按照系统风险评分来调整仓位，历史上比直接持有效果如何？")
        days_back = st.slider("回测时间范围（天）", 30, 180, 60)

        if st.button("开始回测"):
            with st.spinner("正在模拟历史决策，请稍候..."):
                metrics = run_backtest(ticker, days_back)
                st.success(f"{ticker} 回测完成")

                res_p = Path("data/backtest_results.csv")
                if res_p.exists():
                    res_df = pd.read_csv(res_p)
                    res_df["ts"] = pd.to_datetime(res_df["ts"])

                    st.write("### 资金走势对比")
                    st.caption("从同一起点出发，1元钱分别按两种方式操作后的变化")
                    st.line_chart(
                        res_df.set_index("ts")[["equity_hold", "equity_strat"]].rename(
                            columns={"equity_hold": "直接持有", "equity_strat": "AI调仓策略"}
                        )
                    )

                    st.write("### 结果对比")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("AI策略最终收益",   f"{metrics['ret_strat']:+.2%}")
                    c2.metric("直接持有最终收益", f"{metrics['ret_hold']:+.2%}")
                    c3.metric(
                        "AI策略最大亏损幅度",
                        f"{metrics['mdd_strat']:.2%}",
                        delta=f"{metrics['mdd_strat'] - metrics['mdd_hold']:.2%}",
                        help="最大亏损幅度越小说明系统越能帮你控制风险",
                    )
                    c4.metric("直接持有最大亏损幅度", f"{metrics['mdd_hold']:.2%}")

                    st.write("### 期间风险评分走势")
                    st.caption("评分越高系统越倾向于减仓保护，评分越低越倾向于持仓或加仓")
                    st.line_chart(res_df.set_index("ts")[["score"]].rename(columns={"score": "风险评分"}))
                else:
                    st.error("回测数据生成失败，请检查是否有足够历史数据。")


if __name__ == "__main__":
    main()