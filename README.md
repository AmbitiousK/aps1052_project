# aps1052_project

APS1052 课程项目 —— 基于 **BTC（比特币）** 多源数据的量化建模。

## 数据

BTC 相关数据位于 [`raw_data/`](raw_data/)，来源为本人的 HELM 财富引擎（只读导出，未改动源）。详细目录说明、字段格式见 **[raw_data/README.md](raw_data/README.md)**。

### ⚠️ GitHub 上的是「精简安全子集」

完整数据集约 **4.8 GB**，超出 GitHub 限制（单文件 >100MB 拒收、仓库建议 <1GB），因此本仓库**仅收录 GitHub 友好的子集**：

| 已收录（在仓库中） | 未收录（仅本地 `raw_data/`） |
|---|---|
| `01_processed/` 建模就绪特征/标签/切分 | `02_raw_feeds/klines/` 1分钟K线（877M，含超限文件） |
| `02_raw_feeds/` 中小体积 feed（资金费率/基差/估值/期货指标/DVOL/期权OI/COT/链上/净主动量/成交量/OI历史） | `03_unzipped_feeds/` 解压冗余副本（1.0G） |
| `06_source_bundles/` 原始数据包 + NTV建模输出 + 链上指标 | `04_download_klines/` K线下载归档（266M） |
| `07`–`09` 模式挖掘候选 / 决策轨迹 / 期货OI live | `05_realtime/` 实时 tick（逐笔+盘口，2.3G） |
| | `10_wrapped_btc_wbtc/` WBTC（171M） |

被排除的部分可从原始来源重新下载（脚本见 `raw_data/06_source_bundles/` 内各 `*.py`），或向作者索取完整本地副本。

## 建模数据管道

- **[`scripts/build_dataset.py`](scripts/build_dataset.py)** —— 端到端管道：清洗 1 分钟 K 线 / 期货 OI / 资金费率 → 10 个基础可解释的小时级特征 → 未来 1 小时三分类标签（阈值可配，默认路径触及 ±3%）→ `data/processed/btc_1h_dataset_*.parquet`。
- **[`docs/feature_selection.md`](docs/feature_selection.md)** —— 特征选择理由、标签阈值的统计依据、清洗与防泄漏规则、建模指引。
- **[`docs/project_plan.md`](docs/project_plan.md)** —— 项目最终方案：冻结的数据结构、监督学习模型、交易信号与回测框架、Monte Carlo / White Reality Check 统计检验、实验顺序与交付物清单。

```bash
pip install pandas pyarrow numpy
python scripts/build_dataset.py            # 默认 path 标签, ±3%
python scripts/build_dataset.py --threshold 0.01   # 对照实验
```

> 注意：管道需要完整本地数据（1 分钟 K 线不在 GitHub 子集内）。
