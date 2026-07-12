# APS1052 Project Final Plan
## Hourly Extreme-Movement Prediction and Trading Validation

## 1. Project Objective

本项目研究以下问题：

> 在每个整点时刻，利用当时已经可以获得的市场信息，预测未来一小时内价格是否会首先向上突破指定阈值、向下突破指定阈值，或者保持在阈值区间内；随后检验该预测是否具有分类价值、交易价值以及统计显著性。

模型输出为三分类：

$$y_t \in \{-1,\,0,\,+1\}$$

其中：

* $+1$：未来一小时内，价格首先触及上方 barrier；
* $-1$：未来一小时内，价格首先触及下方 barrier；
* $0$：未来一小时内，上下 barrier 均未被触及。

项目的核心不是重新构建数据，而是基于已经完成的数据集，依次完成：

$$\text{Existing Data} \rightarrow \text{Supervised Learning} \rightarrow \text{Trading Signals} \rightarrow \text{Backtesting} \rightarrow \text{Statistical Validation}$$

---

# 2. Data Construction Is Frozen

## 2.1 Existing data pipeline is the project baseline

当前数据构建方案正式冻结，后续不再修改其基本结构。

现有数据设计包括：

* 以整点作为 decision timestamp；
* 每个样本对应未来一小时预测窗口；
* 使用 1-minute K-line 数据判断未来路径；
* 通过 first-touch 规则生成三分类标签；
* 使用小时级特征作为模型输入；
* Open Interest、funding rate 等变量按照实际可获得时间对齐；
* 当前已经生成约 50,522 个 hourly observations；
* 当前使用 10 个具有明确经济含义的输入特征；
* 相邻样本的未来预测窗口不重叠。

后续模型、回测与统计检验必须直接使用该数据集。

除非发现明确的数据泄漏、标签错误或时间对齐错误，否则不得重构数据 pipeline。

---

## 2.2 Decision frequency remains hourly

项目继续使用：

$$\text{Decision Frequency} = 1\ \text{hour}$$

不改为 5-minute decision points。

原因如下：

第一，当前 pipeline 已经使用 1-minute bars 判断未来一小时内的 first-touch，因此路径分辨率已经足够细。

第二，hourly decision timestamps 使相邻样本的未来一小时窗口互不重叠。

如果改为每 5 分钟生成一个样本，相邻样本将共享绝大多数未来价格路径，导致：

* 样本高度相关；
* 同一次市场事件被重复计数；
* 分类指标虚高；
* 有效独立样本数量被夸大；
* 回测中出现大量重叠仓位；
* 统计显著性检验失真。

因此：

> 1-minute bars 用于路径判断，hourly timestamps 用于模型决策。

这是当前项目最合理的数据结构。

---

## 2.3 Existing feature set remains unchanged

当前 10 个特征已经覆盖主要信息维度，包括：

* price return；
* trend；
* volatility；
* trading volume；
* market sentiment；
* leverage and derivatives information。

现阶段不增加大量多窗口特征，也不加入高度相关的技术指标组合。

尤其不引入以下类型的特征膨胀：

* 大量不同窗口的 return；
* 大量不同窗口的 volatility；
* 多组表达相同趋势信息的 moving averages；
* RSI、MACD 与已有 momentum 特征的重复组合；
* 多个相乘形成但缺乏清晰经济意义的 composite indicators。

特征设计继续遵守以下原则：

1. 每个特征必须可以用一句话解释；
2. 每个特征必须只使用当前时刻及以前的信息；
3. 特征之间不应存在严重的信息重复；
4. 模型性能不依赖难以解释的 feature engineering；
5. 项目的贡献重点是完整验证，而不是特征数量。

如果课程明确要求 custom indicator，只在现有特征之外增加最多 1–2 个可解释指标，并作为补充实验，而不是重构现有 feature set。

---

# 3. Label Design

## 3.1 Existing first-touch label mechanism remains unchanged

对于整点时刻 $t$，定义当前价格为 $P_t$，设置 barrier threshold $b>0$。

上方 barrier 为：

$$U_t = P_t(1+b)$$

下方 barrier 为：

$$L_t = P_t(1-b)$$

检查未来一小时内的 1-minute price path，标签定义为：

$$y_t = \begin{cases} +1, & \text{upper barrier is touched first} \\ -1, & \text{lower barrier is touched first} \\ 0, & \text{neither barrier is touched} \end{cases}$$

这一标签定义同时包含：

* direction；
* magnitude；
* path ordering；
* fixed prediction horizon。

它比只看一小时结束价格更符合极端市场信号预测的目标。

---

## 3.2 Barrier threshold is an experimental parameter, not a data redesign

阈值选择属于实验层，而不是重新构建数据逻辑。

建议研究：

$$b \in \{1\%,\ 2\%,\ 3\%\}$$

不同 threshold 使用同一套：

* hourly timestamps；
* 1-minute first-touch；
* feature matrix；
* chronological split；
* evaluation pipeline。

只重新生成对应标签，不更改原始数据结构。

阈值实验的目的包括：

* 比较不同极端程度下的 class distribution；
* 检查极端事件数量是否足够；
* 评估不同 threshold 下的模型可学习性；
* 研究 signal frequency 和 profitability 的关系；
* 防止只选择一个偶然表现最好的 barrier。

主阈值应依据现有数据中的 class distribution、validation performance 和统计稳定性决定。

Test set 不用于选择主阈值。

---

## 3.3 Barrier and trading execution must be aligned

主回测中，label barrier 和交易 TP/SL 必须一致。

如果标签使用 $b=2\%$，那么交易执行应使用：

* take-profit：2%；
* stop-loss：2%；
* maximum holding period：1 hour。

如果标签使用 $b=3\%$，那么交易执行应使用：

* take-profit：3%；
* stop-loss：3%；
* maximum holding period：1 hour。

这样，模型预测目标和实际交易动作完全一致：

> 模型预测未来一小时内价格更可能先触及哪一侧 barrier，交易系统就按照相同 barrier 执行。

非对称 TP/SL 不作为主实验。例如 TP=5%、SL=2% 与对称 ±3% 标签不一致，因为模型没有被训练去预测该非对称交易结果。

非对称 TP/SL 只能作为后续 sensitivity analysis，并且必须明确标注：

> 该实验研究的是 execution rule sensitivity，而不是与标签完全一致的主策略。

---

# 4. Data Integrity Verification

虽然数据层不再重构，但在建模前应完成一次最终审计。

## 4.1 Timestamp audit

确认每个样本满足：

$$\text{feature time} \leq t \quad\text{且}\quad \text{label information} > t$$

需要检查：

* 所有 rolling features 是否在 $t$ 时刻已经可用；
* rolling window 是否只向后看；
* funding rate 是否使用真实发布时间；
* Open Interest 是否没有使用未来更新值；
* normalization 是否没有在完整数据上 fit；
* 标签路径是否严格从 $t$ 之后开始。

---

## 4.2 Missing-value audit

记录：

* 每个特征的 missing count；
* missing value 的来源；
* 删除或填补策略；
* 最终保留样本数量；
* train、validation、test 中的 missing 分布。

处理原则：

* 不使用未来值填补过去数据；
* 不进行 backward fill；
* forward fill 仅用于在经济意义上合理的变量；
* 所有 preprocessing rule 只根据 training data 确定。

---

## 4.3 Class-distribution audit

对每个 barrier threshold 输出：

| Split      | Class -1 | Class 0 | Class +1 | Extreme Ratio |
| ---------- | -------: | ------: | -------: | ------------: |
| Train      |          |         |          |               |
| Validation |          |         |          |               |
| Test       |          |         |          |               |

需要重点观察：

* $+1$ 和 $-1$ 是否极度稀少；
* 三个时间区间的 class distribution 是否稳定；
* test set 是否有足够极端事件；
* 是否存在明显的 market-regime shift。

---

# 5. Chronological Data Split

不得随机打乱整个数据集后进行 train-test split。

使用 chronological split：

$$\text{Train} \rightarrow \text{Validation} \rightarrow \text{Test}$$

一个可行起点是：

* Train：前 60%；
* Validation：中间 20%；
* Test：最后 20%。

或者：

* Train：前 70%；
* Validation：中间 15%；
* Test：最后 15%。

最终比例可以根据事件数量决定，但必须满足：

1. 顺序不被打乱；
2. test period 是最后一段真实未来数据；
3. test 在模型和策略完全冻结前不参与选择；
4. 所有 scaler 仅在 train 上 fit；
5. hyperparameter tuning 仅使用 train 和 validation；
6. confidence threshold 仅使用 validation；
7. test set 只用于最终一次评估。

---

# 6. Exploratory Data Analysis

EDA 只分析现有数据，不重新生成新数据结构。

## 6.1 Required descriptive analysis

输出：

* 每个特征的 summary statistics；
* 每个特征的 distribution；
* class distribution；
* feature correlation matrix；
* 不同标签下的 feature distribution；
* 特征随时间的变化；
* 极端事件的时间聚集情况；
* train、validation、test 的分布差异。

---

## 6.2 Feature redundancy check

虽然现有特征已经经过筛选，仍可计算：

* Pearson correlation；
* Spearman correlation；
* variance inflation factor，若适用；
* mutual information，作为补充。

其目的不是再次大规模删特征，而是证明：

> 当前 10 个特征没有明显的无意义膨胀，并覆盖互补的信息维度。

---

## 6.3 Regime awareness

可以按照时间或 volatility regime 分析：

* high-volatility period；
* low-volatility period；
* bullish period；
* bearish period；
* high-leverage period；
* normal-leverage period。

但 regime 只用于分析模型表现，不修改主数据结构。

---

# 7. Supervised Learning Models

项目不需要大量模型。建议使用一个清晰的模型层级。

## 7.1 Baseline models

### Baseline 1: Majority-class classifier

始终预测最常见类别，通常为 $\hat y_t = 0$。

作用：

* 检查 accuracy 是否被 flat class 主导；
* 提供最低分类基线。

### Baseline 2: Random prediction based on class priors

按照 training class distribution 随机预测。

作用：

* 给出随机分类基准；
* 配合 Monte Carlo 分析。

### Baseline 3: Simple momentum rule

例如根据现有 momentum 或 return 特征构建简单规则：

$$\hat y_t = \begin{cases} +1, & x_t > \theta \\ -1, & x_t < -\theta \\ 0, & \text{otherwise} \end{cases}$$

作用：

* 比较机器学习模型是否优于简单技术规则。

---

## 7.2 Primary models

建议主模型控制在以下范围：

### Multinomial Logistic Regression

作用：

* 线性可解释基线；
* 检查当前特征是否具备简单线性预测能力；
* 分析 coefficient direction。

### Random Forest

作用：

* 学习非线性关系；
* 捕捉 feature interaction；
* 对 scaling 不敏感；
* 提供 feature importance。

### Gradient Boosting Model

可以选择一个：XGBoost / LightGBM / CatBoost / HistGradientBoostingClassifier。只需要选择其中一种作为主要非线性模型。

它通常适合：

* structured tabular data；
* nonlinear effects；
* limited feature count；
* class imbalance。

### Optional Neural Network

神经网络只作为可选对照，不作为必须项。由于现有数据规模和特征数量有限，简单 MLP 即可：

* input：10 features；
* 1–2 hidden layers；
* small number of neurons；
* softmax output；
* early stopping。

不应构建复杂深度架构。

---

# 8. Class-Imbalance Handling

三分类标签很可能由 class 0 主导。需要比较以下处理方式：

## 8.1 No rebalancing

保留原始 class distribution，作为真实数据基线。

## 8.2 Class weights

在训练损失中提高 $-1$ 和 $+1$ 类的权重。例如：

$$w_c = \frac{N}{K\, N_c}$$

其中 $N$ 为总训练样本数，$K=3$ 为类别数量，$N_c$ 为类别 $c$ 的样本数。

## 8.3 Threshold adjustment

不强制通过 oversampling 改变时间序列，而是通过预测概率阈值控制交易信号。

## 8.4 Oversampling caution

不建议直接对时间序列进行普通 SMOTE，因为：

* 它会生成不存在的市场状态；
* 可能破坏时间结构；
* 可能引入不合理的 synthetic extreme events。

如课程要求比较 sampling methods，只能在 training set 内执行，并作为补充实验。

---

# 9. Model Evaluation

## 9.1 Accuracy cannot be the only metric

由于 flat class 可能占多数，overall accuracy 可能具有误导性。必须报告：

* accuracy；
* balanced accuracy；
* macro precision；
* macro recall；
* macro F1；
* weighted F1；
* Matthews correlation coefficient；
* confusion matrix；
* per-class precision；
* per-class recall；
* per-class F1。

---

## 9.2 Extreme-event metrics

重点报告 $\text{Recall}_{+1}$、$\text{Recall}_{-1}$ 以及 $\text{Precision}_{+1}$、$\text{Precision}_{-1}$，因为项目的实际目标不是预测平静市场，而是识别显著上涨和下跌信号。

还可以合并极端类别：

$$z_t = \begin{cases} 1, & y_t \in \{-1,+1\} \\ 0, & y_t = 0 \end{cases}$$

然后评估：

* extreme-event precision；
* extreme-event recall；
* extreme-event F1；
* PR-AUC。

---

## 9.3 Probability quality

如果模型输出概率，应检查：

* log loss；
* Brier score；
* calibration curve；
* reliability diagram。

因为后续交易策略会使用预测 confidence，而不仅是 argmax class。

---

# 10. Trading Signal Construction

## 10.1 Basic signal mapping

将模型预测转为交易动作：

$$a_t = \begin{cases} +1, & \hat y_t = +1 \\ -1, & \hat y_t = -1 \\ 0, & \hat y_t = 0 \end{cases}$$

其中 $+1$ 为 long，$-1$ 为 short，$0$ 为 flat。

---

## 10.2 Confidence-filtered trading

不一定对所有非零预测都交易。设模型输出 $P_t(-1),\,P_t(0),\,P_t(+1)$，只有当极端类别概率超过阈值 $\tau$ 时才开仓：

$$a_t = \begin{cases} +1, & P_t(+1) \geq \tau \\ -1, & P_t(-1) \geq \tau \\ 0, & \text{otherwise} \end{cases}$$

可以在 validation set 比较 $\tau \in \{0.40,\ 0.50,\ 0.60,\ 0.70\}$。

阈值越高：

* 交易次数越少；
* 信号平均置信度越高；
* turnover 越低；
* precision 可能提高；
* recall 可能下降。

最终 threshold 必须在 test 之前冻结。

---

## 10.3 Optional directional asymmetry

也可以分别设置 $\tau_{long}$ 和 $\tau_{short}$，因为上涨和下跌信号的可预测性可能不同。

但这属于补充实验。主策略优先使用统一阈值，以减少参数数量和 data snooping。

---

# 11. Backtesting Framework

## 11.1 Entry

在整点 $t$ 形成信号后，以明确的可执行价格进入。必须说明使用：

* 当前 hourly close；
* 下一分钟 open；
* 或其他没有 look-ahead bias 的 entry price。

最稳妥的设计通常是：

> 使用整点信号生成后的下一分钟 open 作为 entry price。

这样可以避免假设模型能够在同一个 closing timestamp 无延迟成交。如果当前数据定义已经明确 entry price，则保持原定义，不重新改变 pipeline。

---

## 11.2 Position rules

每个整点最多形成一个新仓位。主回测规则：

* signal $+1$：开 long；
* signal $-1$：开 short；
* signal $0$：不开仓；
* 最大持仓时间：1 hour；
* 使用 1-minute path 判断 TP/SL first-touch；
* 触及 barrier 时立即平仓；
* 一小时内没有触及 barrier，则 time-stop 平仓；
* 同一时间不持有多个重叠仓位。

由于决策频率为 1 hour，该设置天然与 non-overlapping prediction windows 一致。

---

## 11.3 Long return

对于 long position：

$$r_t^{long} = \frac{P_{exit}-P_{entry}}{P_{entry}}$$

---

## 11.4 Short return

对于 short position：

$$r_t^{short} = \frac{P_{entry}-P_{exit}}{P_{entry}}$$

---

## 11.5 Transaction costs

净收益为：

$$r_t^{net} = r_t^{gross} - c_{entry} - c_{exit} - s_{entry} - s_{exit}$$

其中 $c$ 为 fee，$s$ 为 slippage。

建议至少设置三种成本情景：

### Zero-cost scenario

用于展示模型预测的理论上限，但不作为主要结果。

### Base-cost scenario

使用合理的交易费用和滑点，作为主结果。

### High-cost scenario

用于测试策略在更差执行环境下是否仍然稳定。

成本数值应根据所使用市场和交易平台进行说明。

---

## 11.6 Position sizing

主实验采用固定仓位 $\text{position size} = 1$，或者每次使用固定比例资本 $w_t = w$。

不要在主实验中加入：

* leverage optimization；
* Kelly criterion；
* volatility targeting；
* dynamic position sizing；
* pyramiding。

这些会使收益来源难以归因。

---

# 12. Trading Metrics

分类指标和交易指标必须分开报告。

## 12.1 Return metrics

* total return；
* cumulative return；
* annualized return，若时间尺度适合；
* average return per trade；
* median return per trade。

## 12.2 Risk-adjusted metrics

### Sharpe ratio

$$\text{Sharpe} = \frac{E[R_t - R_f]}{\sigma(R_t - R_f)}$$

对于短周期 crypto strategy，可设 $R_f = 0$，但需要说明。

### Sortino ratio

$$\text{Sortino} = \frac{E[R_t - R_f]}{\sigma_{downside}}$$

### Maximum drawdown

$$\text{MDD} = \max_t \left( \frac{\text{Peak}_t - \text{Equity}_t}{\text{Peak}_t} \right)$$

---

## 12.3 Trade-quality metrics

* number of trades；
* win rate；
* loss rate；
* average win；
* average loss；
* payoff ratio；
* profit factor；
* average holding period；
* long trade count；
* short trade count；
* long profitability；
* short profitability；
* turnover。

Profit factor 定义为：

$$\text{Profit Factor} = \frac{\text{Gross Profit}}{|\text{Gross Loss}|}$$

---

# 13. Equity Curve Evaluation

课程要求下，应分别展示：

* train equity curve；
* validation equity curve；
* test equity curve。

但需要明确：

* train curve 只用于诊断；
* validation curve 用于策略与阈值选择；
* test curve 用于最终 out-of-sample 结论。

建议同时展示：

1. gross equity curve；
2. net equity curve；
3. benchmark curve；
4. drawdown curve。

可以选择的 benchmark：buy-and-hold / always-flat / random signal strategy / simple momentum strategy。

---

# 14. Sensitivity Analysis

敏感性分析必须围绕现有数据开展，而不是重新设计数据。

## 14.1 Barrier sensitivity

比较 $b = 1\%,\ 2\%,\ 3\%$，观察：

* class distribution；
* classification metrics；
* signal count；
* trade count；
* net return；
* Sharpe；
* max drawdown；
* statistical significance。

---

## 14.2 Confidence-threshold sensitivity

比较 $\tau = 0.40,\ 0.50,\ 0.60,\ 0.70$，观察：

* coverage；
* precision；
* recall；
* number of trades；
* turnover；
* profitability。

---

## 14.3 Transaction-cost sensitivity

比较不同费用和滑点假设。用于回答：

> 策略利润是否来自真实预测能力，还是会被轻微交易成本完全消除？

---

## 14.4 Model sensitivity

比较 logistic regression / random forest / gradient boosting / optional neural network。

重点是模型结论是否一致，而不是只展示表现最好的模型。

---

## 14.5 TP/SL sensitivity

主回测保持 barrier-aligned TP/SL。补充实验可以比较：

* symmetric TP/SL；
* slightly asymmetric TP/SL；
* time-stop only；
* barrier exit only。

但必须清楚区分：

* 主实验：预测目标和执行规则一致；
* 补充实验：执行规则 sensitivity。

---

# 15. Monte Carlo Permutation Test

## 15.1 Purpose

检验观察到的交易表现是否可能来自随机信号。原假设：

$$H_0: \text{模型信号与未来收益之间没有真实关系}$$

---

## 15.2 Basic procedure

保持真实市场 returns 或 price paths 不变。随机打乱：

* prediction labels；
* trading signals；
* 或 signal-time mapping。

对于每次 permutation：

1. 生成随机化信号；
2. 使用相同 backtest engine；
3. 使用相同交易成本；
4. 计算目标统计量。

重复 $B = 1000$ 或更多次。

---

## 15.3 Test statistics

可选统计量包括：total net return / Sharpe ratio / profit factor / maximum drawdown adjusted return。

主统计量建议提前指定，避免事后选择。

---

## 15.4 Empirical p-value

$$p = \frac{1 + \sum_{b=1}^{B} I(T_b \geq T_{obs})}{B + 1}$$

如果 $p$ 较小，说明真实模型策略表现显著优于随机信号。

---

## 15.5 Time-series caution

不应简单破坏全部时间结构。可以考虑：

* block permutation；
* block bootstrap；
* circular block shift。

具体方法取决于课程要求，但原则是尽可能保留市场收益的时间依赖结构。

---

# 16. White's Reality Check

## 16.1 Purpose

项目会比较多个模型、多个 barrier、多个 confidence threshold、多个 cost assumptions、多个 strategy variations。

如果只选择其中表现最好的策略，就会产生 data-snooping bias。White's Reality Check 用于检验：

> 最佳策略是否真的优于 benchmark，还是仅仅因为尝试了很多候选组合而偶然胜出。

---

## 16.2 Candidate strategy set

必须提前定义 candidate family，例如：

$$\mathcal{M} = \{\text{model} \times \text{barrier} \times \text{confidence threshold}\}$$

应控制 candidate 数量，避免无止境参数搜索。

---

## 16.3 Benchmark

可选 benchmark：zero-return strategy / always-flat / buy-and-hold / simple momentum strategy。需要在报告中明确说明为什么选择该 benchmark。

---

## 16.4 Bootstrap

由于收益是时间序列，应使用：stationary bootstrap / moving-block bootstrap / circular block bootstrap。普通 iid bootstrap 可能破坏收益自相关。

---

## 16.5 Interpretation

如果 Reality Check 仍然拒绝原假设，则可以更有把握地说明：

> 在考虑多模型与多参数搜索以后，最佳策略仍然显示出超越 benchmark 的统计证据。

如果不能拒绝，也不代表项目失败。正确结论可能是：

> 某些策略在样本外表现较好，但在校正 data snooping 后，证据不足以确认其具有稳定超额收益。

这同样是严谨且有价值的研究结果。

---

# 17. Model Interpretation

## 17.1 Logistic regression coefficients

分析：

* 哪些特征提高 $+1$ 概率；
* 哪些特征提高 $-1$ 概率；
* 哪些特征与 flat class 相关。

## 17.2 Tree-based importance

可以使用 impurity importance / permutation importance / SHAP（若时间允许）。

优先使用 permutation importance，因为普通 tree importance 可能偏向连续变量或高方差变量。

## 17.3 Event-level explanation

选择若干真实 test events：

* correctly predicted upward breakout；
* correctly predicted downward breakout；
* false positive；
* missed extreme event。

展示：当时特征状态 / 模型概率 / 未来一小时路径 / barrier first-touch / 最终交易结果。

这样能够把统计结果和真实市场行为连接起来。

---

# 18. Ablation Study

在不改变数据构建的前提下，可以按现有信息维度做特征组 ablation。例如分别删除：

* price/trend features；
* volatility features；
* volume features；
* sentiment features；
* leverage and derivatives features。

比较：macro F1 / extreme recall / PR-AUC / Sharpe / net return。

目的不是重新设计特征，而是回答：

> 哪类信息对极端行情预测贡献最大？

---

# 19. Robustness Checks

## 19.1 Long and short separately

分别评估 upward breakout prediction / downward breakout prediction / long strategy / short strategy。

可能出现：

* 下跌信号更容易预测；
* 做空信号更少但 precision 更高；
* 做多策略在 bull regime 中表现更好。

这些不应被 overall metrics 掩盖。

## 19.2 Market-regime robustness

检查模型在 high-volatility / low-volatility / bullish / bearish regime 中的表现。

## 19.3 Time stability

将 test period 分为若干连续区间，观察：

* 是否只有某一个月份赚钱；
* 是否存在单次极端事件主导收益；
* 是否随时间快速衰减。

## 19.4 Remove-best-trades test

删除表现最好的若干笔交易后重新计算收益。例如删除最佳 1 笔 / 最佳 3 笔 / 最佳 5 笔。

用于检查策略是否过度依赖少数幸运事件。

---

# 20. Recommended Experimental Order

## Stage 1: Freeze and audit data

完成：data dictionary / timestamp audit / missing-value audit / class distribution / chronological split / feature correlation analysis。不更改 hourly pipeline。

## Stage 2: Build classification baselines

完成：majority baseline / random baseline / momentum baseline / logistic regression。

## Stage 3: Train nonlinear models

完成：random forest / gradient boosting / optional MLP。

## Stage 4: Select model using validation set

依据：macro F1 / balanced accuracy / extreme precision-recall / probability calibration / validation trading performance。

## Stage 5: Construct trading signals

完成：long/short/flat / confidence filtering / fixed position sizing。

## Stage 6: Build aligned backtest

完成：barrier-aligned TP/SL / time stop / 1-minute first-touch / fees / slippage / equity curves / financial metrics。

## Stage 7: Statistical validation

完成：permutation test / bootstrap confidence intervals / White Reality Check。

## Stage 8: Final out-of-sample test

冻结：selected model / hyperparameters / barrier / confidence threshold / trading cost / execution rules。然后只运行一次 final test。

## Stage 9: Produce final report and slides

将结果组织成：problem / data / labels / features / models / classification results / trading results / statistical tests / limitations / conclusion。

---

# 21. Deliverables

最终提交材料建议包括：

## Notebook

至少包含：

1. data loading；
2. data audit；
3. EDA；
4. chronological split；
5. preprocessing；
6. baseline models；
7. primary models；
8. classification evaluation；
9. signal generation；
10. backtesting；
11. financial metrics；
12. permutation test；
13. White Reality Check；
14. robustness analysis；
15. final figures and tables。

## Data files

* final modelling CSV；
* data dictionary；
* feature-definition table；
* label-definition table；
* model predictions；
* trade ledger；
* equity curve outputs；
* statistical-test outputs。

## Environment

* conda package list；
* requirements file；
* random seed configuration；
* Python version；
* library versions。

## Slides（约 25–30 页）

1. Title；
2. Motivation；
3. Research question；
4. Data sources；
5. Existing hourly data architecture；
6. Why hourly decision points；
7. First-touch label；
8. Feature set；
9. Leakage prevention；
10. Class distribution；
11. Time split；
12. EDA；
13. Baselines；
14. Models；
15. Classification metrics；
16. Confusion matrix；
17. Extreme-event evaluation；
18. Signal construction；
19. Backtest rules；
20. Transaction costs；
21. Train equity curve；
22. Validation equity curve；
23. Test equity curve；
24. Financial metrics；
25. Barrier sensitivity；
26. Cost sensitivity；
27. Permutation test；
28. White Reality Check；
29. Limitations；
30. Conclusion。

---

# 22. Final Project Positioning

本项目不是重新构造一个更高频、更复杂的数据集。它的核心优势是：

1. 使用整点决策，避免未来窗口重叠；
2. 使用 1-minute price path 判断 first-touch；
3. 特征均来自决策时刻已经可获得的信息；
4. 特征数量有限、含义清晰；
5. 标签同时表示方向、幅度与路径顺序；
6. 分类预测与交易执行规则保持一致；
7. 同时检验分类性能、经济收益和统计显著性；
8. 通过 Reality Check 控制 data-snooping bias。

最终研究问题应表述为：

> Can interpretable market, sentiment, and leverage features observed at hourly decision points predict which price barrier will be reached first during the following hour, and can such predictions generate statistically and economically meaningful trading performance after transaction costs?

整个后续工作都建立在现有 data construction 之上。当前数据 pipeline 是项目基础，不是需要被替换的部分。
