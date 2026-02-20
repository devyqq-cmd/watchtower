# Watchtower

基于 Python 的轻量级行情监控：使用 **yfinance** 拉取小时级别数据，存入 **SQLite**，并通过 **Streamlit** 展示。

支持标的：

- `0700.HK`
- `COIN`
- `INFQ`

默认时间级别：**1 小时（60m）**

## 安装

在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用: .venv\Scripts\activate
pip install -e ".[dev]"
```

## 配置

- **全局配置**：`config.yaml`
  - `tickers`: 监控的标的列表
  - `interval`: yfinance 时间间隔（已设置为 `60m`）
  - `days`: 初始拉取天数
  - `db_path`: 默认 SQLite 数据库路径（可被环境变量覆盖）

- **环境变量模板**：`.env.example`
  - 可复制为 `.env` 并修改：

```bash
cp .env.example .env
```

支持的环境变量：

- `WATCHTOWER_DB_PATH`：覆盖默认数据库路径

## 数据采集

在项目根目录运行：

```bash
python -m jobs.ingest
```

行为：

- 使用 `config.yaml` 中的配置和 `WATCHTOWER_DB_PATH`（如果存在）
- 调用 `yfinance.download` 拉取 1 小时 K 线
- 将数据写入 SQLite 表 `prices`，去重键为 `(ticker, ts)`

## Streamlit 面板

在项目根目录运行：

```bash
streamlit run app/streamlit_app.py
```

功能：

- 自动检测数据库中已存在的 `ticker`
- 按日期过滤，展示收盘价曲线
- 查看最近 K 线表格和基础统计信息

## 测试

使用 `pytest`：

```bash
pytest
```

## GitHub Actions CI

工作流文件：`.github/workflows/ci.yml`

主要步骤：

- 使用 Python 3.11
- 安装依赖：`pip install -e ".[dev]"`
- 运行测试：`pytest`

