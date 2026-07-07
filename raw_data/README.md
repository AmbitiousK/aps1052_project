# BTC 数据集（raw_data）

本文件夹汇总了从 **HELM（helm_wealth）** 财富引擎中导出的、与 **BTC（比特币）** 相关的**全部数据**，用于 APS1052 课程项目。

> **来源**：`~/Desktop/helm_wealth/engine/`（只读复制，**未对 HELM 源做任何改动**）
> **导出方式**：`rsync`/`cp` 只读镜像；已排除 `__pycache__`、`.pyc`、`.DS_Store` 和密钥文件（`.env`）。
> **范围**：HELM 中所有 BTC 数据的**全量镜像**（含原始 feed、解压副本、下载源、实时 tick、建模产物）。
> **导出日期**：2026-07-06

---

## 目录总览

| 目录 | 内容 | 来源（HELM 内路径） |
|---|---|---|
| `01_processed/` | 建模就绪产物：特征矩阵、标签、切分、6h 蜡烛 | `engine/processed/*/btc` |
| `02_raw_feeds/` | 各类原始下载 feed（K线/资金费率/基差/估值/期货指标/波动率/期权OI/COT/链上/净主动量/成交量统计/期货OI历史） | `engine/raw/<feed>/btc` |
| `03_unzipped_feeds/` | `02` 中压缩包**解压后的副本**（内容与 raw 对应，便于直接读取） | `engine/unzip/<feed>/btc` |
| `04_download_klines/` | K线下载源（Binance/Bitstamp 原始归档） | `engine/download/klines/btc` |
| `05_realtime/` | 实时 tick 级数据：逐笔成交、盘口 BBO、实时K线、净主动量（多交易所） | `engine/realtime/store/<type>/btc` |
| `06_source_bundles/` | 项目最初的原始数据包与建模输出（**仅保留 BTC 相关文件**） | `engine/unhandle_data/` |
| `07_research_candidates/` | 模式挖掘（pattern mining）候选信号 | `engine/research/pattern_mining/store/candidates_btc*` |
| `08_decision_traces/` | 策略决策轨迹快照（历史 + 最新 + 摘要 + resident packet） | `engine/scheduler/decision_trace_store/btc*` |
| `09_futures_oi_live/` | 期货未平仓合约（OI）实时序列 | `engine/store/futures_oi/live/btc.csv` |
| `10_wrapped_btc_wbtc/` | **WBTC（Wrapped BTC，包装比特币）** —— 严格来说是另一个代币，非原生 BTC，单独隔离存放 | `engine/*/wbtc` |

---

## 各目录明细

### 01_processed/ —— 建模就绪数据（推荐入口）
- `feature_matrix/btc/` —— 特征矩阵 v1
- `feature_matrix_v2_2/btc/` —— 特征矩阵 v2.2（更新版本）
- `labels/btc/1h/` —— 标签，按月分片（`YYYY-MM.parquet`），1 小时粒度
- `splits/btc/1h/` —— 训练/验证/测试切分（含 `_manifest/` 血缘）
- `btc_6h.parquet` —— 6 小时蜡烛（market candles）

### 02_raw_feeds/ —— 原始 feed
按 `<feed>/btc/<交易所>/<产品-粒度>/` 组织。包含：
- `klines/` —— **1 分钟 K线**（Binance `spot-BTCUSDT-1m` 月/日归档 + Bitstamp `spot-BTCUSD-1m`，2012–2025 合并版）**〔本目录最大项〕**
- `funding_rate/` —— 资金费率（Binance UM 8h）；表头 `funding_time,datetime,symbol,funding_rate,mark_price`
- `basis/` —— 现货-期货基差（Binance spot_fut 1d）
- `valuation/` —— 市值/估值（cryptodatadownload spot 1d）
- `futures_metrics/` —— 期货指标（Binance UM 5m）
- `dvol/` —— Deribit DVOL 隐含波动率指数（1d）
- `options_oi/` —— 期权未平仓合约（Binance options 1d）
- `cot/` —— COT 持仓报告
- `onchain_metrics/` —— 链上指标（BGeometrics spot 1d）
- `onchain_blocks/` —— 链上区块数据（cryptodatadownload chain 1d）
- `net_taker_volume/` —— 净主动成交量（Binance UM 1d）
- `volume_stats/` —— 成交量统计（Binance spot 1d）
- `futures_oi_history/` —— 期货 OI 历史，逐日 `BTCUSDT-metrics-YYYY-MM-DD.zip`（约 2100 个文件）

### 03_unzipped_feeds/ —— 解压副本
与 `02_raw_feeds/` 一一对应，是压缩包解压后的明文版本，可直接读取。（`klines` 解压后体积较大。）

### 04_download_klines/ —— K线下载归档
K线的另一份下载来源（含 `bitstamp-btcusd-minute-data` 等原始归档）。

### 05_realtime/ —— 实时 tick 数据
多交易所（binance / okx / coinbase / bybit / bitget / kraken）：
- `agg_trades/btc/` —— 逐笔聚合成交〔**体积最大**，约 1.9G〕
- `bbo/btc/` —— 最优买卖盘口（Best Bid/Offer）
- `klines/btc/` —— 实时 1m K线
- `net_taker_volume/btc/` —— 实时净主动成交量

### 06_source_bundles/ —— 原始数据包（BTC 部分）
> 原始 Bundle 为混合资产包（含 ETH/BNB/LTC/SOL/XRP 等），此处**仅保留 BTC 相关文件**；共享下载脚本一并保留；**API 密钥文件 `.env` 已排除**。

- `CryptoDataBundle1_btc/` —— BTCUSDT OHLCV+TA、DVOL、市值、Binance 溢价统计、链上区块；`cryptodatadwnld_programs/` 下载与建模脚本
- `CryptoDataBundle2_btc/` —— Binance/Deribit 的 BTC OHLC（日/时）、资金费率、期货指标、成交量、COT（2017–2025）、期权 OI、基差、DVOL；相关下载脚本
- `outputs_btc_ntv_project/` —— BTC「净主动量（NTV）」建模项目产物（回测/系数/特征重要性/模型对比）
- `outputs_btc_ntv_revised/` —— 上述项目的修订版产物
- `BGMetricsIndicators_btc/` —— BGeometrics 链上指标（mvrv_z-score / nupl / nvt_ratio / puell_multiple / sopr）+ 采集脚本 `btc_onchain.py`
- `BitcoinIndicators.txt`、`btc_ohlcv.csv`（日线 OHLCV）、`btc_funding_rate.csv` —— 顶层散文件

### 07_research_candidates/ —— 模式挖掘候选
`candidates_btc_24h.*` / `candidates_btc_multi.*`（csv + parquet 双格式）。

### 08_decision_traces/ —— 决策轨迹
- `btc_history.jsonl` —— 历史决策流水
- `btc_latest.json` —— 最新一条（含 `final_verdict` / `direction` / `confidence` / 候选模式等，schema v0.2）
- `btc_light_summary.json`、`btc_resident_packet.json` —— 摘要 / 常驻打包

### 09_futures_oi_live/
`btc.csv` —— 期货 OI 实时序列；表头 `timestamp,sum_oi,sum_oi_value`。

### 10_wrapped_btc_wbtc/ —— WBTC（隔离）
**WBTC（Wrapped BTC）是与 BTC 1:1 锚定但独立发行的 ERC-20 代币，并非原生 BTC。** 为完整起见一并镜像，但单独存放，请勿与原生 BTC 数据混用：
- `raw_klines/` —— WBTC 原始 1m K线
- `realtime/` —— WBTC 实时 agg_trades / bbo / net_taker_volume

---

## 常见文件格式速查
- **日线 OHLCV**（`btc_ohlcv.csv`）：`date, btc_open, btc_high, btc_low, btc_close, btc_volume`
- **小时 OHLC**（`BTC_Binance_OHLC_hourly.csv`）：`unix, date, symbol, open, high, low, close, volume, volume_from, marketorder_volume, ..., tradecount`
- **资金费率**：`funding_time, datetime, symbol, funding_rate, mark_price`
- **期货 OI live**：`timestamp(ms), sum_oi, sum_oi_value`
- **K线归档**：Binance 官方 `data/spot/{daily,monthly}/klines/BTCUSDT/...` 目录结构；Bitstamp 提供合并 CSV。
- **parquet**：`processed/` 与部分 research 产物为列式 parquet，用 pandas/pyarrow 读取。

---

## 重要说明
1. **HELM 源未被改动** —— 本目录全部为只读复制，任何修改请只在 `raw_data/` 内进行。
2. **已排除敏感文件** —— `.env`（API 密钥）等未纳入。
3. **存在冗余（有意保留）** —— `03_unzipped_feeds/` 是 `02_raw_feeds/` 的解压副本；`04_download_klines/` 与 `02/klines/` 存在来源重叠。这是「全量镜像」的结果，如只需建模数据，优先看 `01_processed/` 与 `06_source_bundles/`。
4. **体积** —— 全量约 5–6 GB，主要集中在 `05_realtime/`（tick 级）与各 `klines/`。
