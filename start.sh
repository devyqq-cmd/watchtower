#!/usr/bin/env bash
# Watchtower 启动脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# 用法说明
usage() {
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  ingest      抓取数据并评估警报（默认）"
    echo "  dashboard   启动 Streamlit 仪表板"
    echo "  backtest    回测（默认股票: 0700.HK）"
    echo "  validate    滚动验证（walk-forward）"
    echo "  sweep       参数敏感性分析"
    echo "  review      周度信号质量回顾"
    echo "  test        运行所有测试"
    echo "  all         依次运行 ingest + dashboard"
    echo ""
    echo "示例:"
    echo "  $0                  # 默认运行 ingest"
    echo "  $0 dashboard        # 启动仪表板"
    echo "  $0 backtest 9988.HK # 回测指定股票"
}

# 检查 uv 是否安装
check_uv() {
    if ! command -v uv &>/dev/null; then
        error "未找到 uv，请先安装: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
}

# 安装依赖
sync_deps() {
    info "同步依赖..."
    uv sync --quiet
}

CMD="${1:-ingest}"
TICKER="${2:-0700.HK}"

check_uv
sync_deps

case "$CMD" in
    ingest)
        info "抓取数据 + 评估警报..."
        uv run python -m jobs.ingest
        ;;
    dashboard)
        info "启动 Streamlit 仪表板（http://localhost:8501）..."
        uv run streamlit run app/streamlit_app.py
        ;;
    backtest)
        info "回测: $TICKER"
        uv run python -m jobs.backtest "$TICKER"
        ;;
    validate)
        info "滚动验证: $TICKER"
        uv run python -m jobs.walk_forward "$TICKER"
        ;;
    sweep)
        info "参数敏感性分析: $TICKER"
        uv run python -m jobs.param_sweep "$TICKER"
        ;;
    review)
        info "周度信号质量回顾..."
        uv run python -m jobs.review
        ;;
    test)
        info "运行所有测试..."
        uv run python -m pytest -v
        ;;
    all)
        info "抓取数据 + 评估警报..."
        uv run python -m jobs.ingest
        info "启动 Streamlit 仪表板（http://localhost:8501）..."
        uv run streamlit run app/streamlit_app.py
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        error "未知命令: $CMD"
        usage
        exit 1
        ;;
esac
