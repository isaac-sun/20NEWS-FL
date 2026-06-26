# 20 Newsgroups Federated Learning — Free-Rider Attack & Shapley Detection

## 环境准备

```bash
cd /Users/isaac/Codes/Researches/20NEWS-FL
pip install -r requirements.txt
```

依赖：`torch`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `openpyxl`, `tqdm`, `transformers`

---

## 实验入口

| 脚本 | 攻击类型 | 说明 |
|------|---------|------|
| `main.py` | DFR / SDFR / AFR | 搭便车攻击（Free-Rider） |

每次运行 **4 组实验**（1 个 baseline + 3 个攻击），结果输出到 `results/`。

---

## 快速开始

```bash
python main.py
```

> **首次运行** 会自动下载：
> - 20 Newsgroups 数据集（~20MB）
> - bert-base-uncased 模型（~440MB）
> - 然后用 frozen BERT 预计算全部文本的 768 维 embedding（约 1-3 分钟）

如果本地已有缓存，后续运行直接读取，不再重新下载。

---

## 配置参数说明

所有参数集中在 `config.py` → `Config` dataclass：

### 数据
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_clients` | 10 | 客户端总数 |
| `iid` | True | True=IID 划分, False=Non-IID |
| `val_ratio` | 0.1 | 验证集占训练集比例 |
| `bert_model_name` | `bert-base-uncased` | HuggingFace 模型名 |

### 模型
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_dim` | 256 | 分类头隐藏层维度 |
| `num_classes` | 20 | 20 Newsgroups 类别数 |

### 联邦学习
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_rounds` | 30 | 通信轮数 |
| `local_epochs` | 3 | 客户端本地训练轮数 |
| `local_lr` | 0.001 | 本地学习率 (Adam) |
| `server_lr` | 1.0 | 服务端聚合步长 |
| `participation_ratio` | 0.8 | 每轮参与客户端比例 |
| `batch_size` | 64 | 训练/评估 batch size |

### 攻击
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `attack_type` | `"none"` | `"none"`, `"dfr"`, `"sdfr"`, `"afr"` |
| `malicious_ratio` | 0.4 | 恶意客户端比例 |
| `dfr_sigma` | 0.5 | DFR 噪声初始标准差 |
| `dfr_gamma` | 1.0 | DFR 噪声衰减指数 |
| `afr_noisy_frac` | 0.1 | AFR 噪声扰动参数比例 |

### Shapley
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_mc_samples` | 30 | Monte Carlo permutation 次数 |
| `utility_alpha` | 0.5 | Utility score EMA 系数 |

---

## 输出结果

### 控制台输出

每个实验结束后打印汇总表格：

```
==============================================================================================================
EXPERIMENT SUMMARY
==============================================================================================================
  baseline_no_attack      0.6893  1.0948  +0.008892  +0.000000  1.29e-03  0.00e+00  0.380465  0.000000
  attack_dfr              0.6693  1.1684  +0.040620  -0.040508  2.98e-03  2.21e-02  0.959273  0.074677
  ...
```

列含义：`Acc` `Loss` `SV_h`(honest mean SV) `SV_m`(malicious mean SV) `Var_h` `Var_m` `PosSum_h` `PosSum_m`

### Excel 文件

`results/experiment_results.xlsx` 包含三个 sheet：
- **round_shapley_details**：每轮每个客户端的详细 SV 和聚合指标
- **experiment_summary**：实验汇总
- **per_class_records**：每轮每个客户端的 20 类 raw Shapley 值

### 图表

所有图表输出到 `results/plots/`，使用 seaborn 专业风格 + 300dpi：

| 图片 | 内容 |
|------|------|
| `fig_01_accuracy_loss.png` | 全局 acc + loss 曲线，含 baseline 对比 |
| `fig_02_shapley_round_cumulative.png` | 左：round SV 折线；右：cumulative SV 柱状 |
| `fig_03_shapley_heatmap.png` | 每客户端每轮 SV 热力图（4 实验），恶意行红框 |
| `fig_04_per_class_fingerprint.png` | 每类 SV 指纹（honest 有峰 vs free-rider 平坦） |
| `fig_05_two_metric_scatter.png` | Variance vs Positive Sum 散点图（后 5 轮均值） |
| `fig_06_metric_boxplots.png` | 上排方差箱线，下排正和箱线 |
| `fig_07_per_class_sv_deepdive.png` | DFR 深度：单诚实 vs 单恶意 class×round 热力图 |
| `fig_08_class_metrics_overview.png` | 方差/正和随轮次折线 + 最终均值柱状 |
| `fig_09_cumulative_sv_bar.png` | 每实验各客户端累积 SV 柱状（颜色区分诚实/恶意） |
| `fig_10_multi_attack_summary.png` | 多攻击并排：方差和正和的诚实 vs 恶意对比 |

---

## Shapley 值计算说明

本项目的 Shapley 值采用 **Monte Carlo permutation sampling**：

1. **Value function**（对类别 c）：
   ```
   v_c(S) = Loss(w_g, D_v^c) - Loss(w_S, D_v^c)
   ```
   含义：coalition S 相比全局模型让类别 c 的验证 loss 降低了多少。

2. **Coalition 模型**（对齐朋友标准）：
   ```
   w_S = Σ_{j∈S} n_j · w_j  /  Σ_{j∈S} n_j
   ```
   即按客户端样本数加权平均的本地模型参数，不使用 `server_lr`。

3. **边际贡献**：
   ```
   Δloss_i,c = previous_loss_c - current_loss_c
   ```
   客户端 i 加入 coalition 后，loss 降低越多 → SV 越大。

4. **Shapley 估计**：
   ```
   SV_i,c ≈ (1/M) Σ_m [ v_c(P_i^m ∪ {i}) - v_c(P_i^m) ]
   ```
   M = `num_mc_samples`（默认 30 次随机排列）

5. **摘要指标**（per client）：
   - `class_sv_variance`：20 个类别 SV 的方差（honest 高 → 贡献有类别特异性）
   - `positive_class_sv_sum`：正 SV 之和（honest 高 → 有真实正向贡献）

---

## 模型架构

- **Frozen BERT** (bert-base-uncased)：预计算 768 维 sentence embedding（不参与训练）
- **Classifier head**：`Linear(768, hidden) → ReLU → Dropout → Linear(hidden, 20)`
- 总参数量：~202K（仅 classifier head 参与 FL 训练）
