# 实验配置与攻击对齐说明

## 当前执行配置

### 数据与模型

| 项目 | 配置 |
|---|---|
| 数据集 | 20 Newsgroups |
| 预处理 | 移除 headers、footers、quotes |
| 划分 | 训练集的 10% 作为分层验证集 |
| 客户端 | 10，IID |
| 模型 | `distilbert-base-uncased`，全参数训练 |
| 分类头 | Dropout(0.3) + Linear(768, 20) |
| 最大序列长度 | 512 |

### 联邦训练

| 参数 | 值 |
|---|---:|
| 通信轮数 | 50 |
| 每轮参与率 | 1.0 |
| 本地 epoch | 2 |
| 本地学习率 | 2e-5 |
| batch size | 8 |
| 优化器 | AdamW |
| weight decay | 0.01 |
| label smoothing | 0.1 |
| warmup ratio | 0.1 |
| 梯度裁剪 | 1.0 |
| 聚合 | 按客户端样本量加权的标准 FedAvg |
| 服务端动量/学习率 | 无 |

### 攻击与 Shapley

| 参数 | 值 |
|---|---:|
| 恶意客户端比例 | 0.4（客户端 0–3） |
| SDFR/AFR honest warm-up | 2 轮 |
| DFR gamma | 1.0 |
| DFR sigma | 由首次全局模型增量自动拟合 |
| AFR C | 由 honest warm-up 更新按 Lemma 1/2 自动估计 |
| AFR 噪声参数比例 | 0.1 |
| Monte Carlo permutations | 15 |
| Shapley eval batch size | 按 GPU profile 自动选择 16 / 32 / 64 |
| seed | 42 |

### NVIDIA GPU profile

训练 batch 始终为 8，保证不同 GPU 上的训练语义一致。`--gpu-profile auto`
只根据显存调整 Shapley 推理 batch 和 DataLoader workers：

| 显存 | profile | eval batch | workers |
|---:|---|---:|---:|
| ≤16GB | t4 | 16 | 2 |
| 16–40GB | medium | 32 | 4 |
| >40GB | large | 64 | 4 |

推荐指令：

```bash
FORCE_DEVICE=cuda python -u main.py --require-cuda --gpu-profile auto
```

`--require-cuda` 会在 CUDA 不可用时立即停止，防止完整实验意外在 CPU 上运行。
`--results-dir PATH` 可修改结果目录。最终解析后的 GPU 名称、显存、profile、
batch 和 workers 会写入 Excel 的 `experiment_config` sheet。

## 攻击公式

### DFR

依据 Fraboni 等人的 disguised free-rider：

```text
delta_f(t) = sigma * t^(-gamma) * epsilon_t
epsilon_t ~ N(0, I)
```

`sigma` 使用第一次可观测的 `theta(1)-theta(0)` 所有坐标的总体标准差。
在该增量尚不可观测的第 1 轮，客户端返回 plain free-rider 的零更新，避免使用
与论文无关的固定大噪声。

### SDFR

依据 Zhu 等人 Eq. (9)：

```text
U_f(theta) = ||theta(t)-theta(t-1)|| / ||theta(t-1)-theta(t-2)||
             * (theta(t)-theta(t-1))
```

### AFR

依据 Zhu 等人 Lemma 1、Lemma 2 和 Algorithm 1：

```text
E[cos beta] = C^2 / (C^2 + exp(2 * lambda_bar * t))

|phi(t)| = sqrt(n^2 / (n + (n^2-n)E[cos beta]) - 1) * |U_f(theta)|

U_hat_f(theta) = U_f(theta) + phi(t) * N(0, 1/d)
```

噪声只施加到比例为 `d/D` 的随机参数坐标。`lambda_bar` 根据全局模型增量
`l(t)=||theta(t)-theta(t-1)||` 的衰减估计。论文 Algorithm 1 排版中的比值会得到
负衰减率，与其 Lemma 1 和 `exp(-lambda*t)` 推导矛盾；实现采用与推导一致的
`-log(l(t)/l(1))/(t-1)`。

## 数值安全

- 客户端上传和全局状态统一存储在 CPU，避免 CUDA/MPS 设备混用。
- 每个客户端更新、聚合状态、模型 logits、loss 和 Shapley coalition 都检查有限值。
- 任意 NaN/Inf 会立即终止实验，不再导出看似成功的结果。
- Excel 包含实际运行配置，结果可以追溯。

## 论文来源

- Jin, Hu, Min, *Robust and Fair Federated Learning Based on Model-Agnostic Shapley Value*, IEEE Transactions on Networking, DOI: 10.1109/TON.2025.3630314.
- Fraboni, Vidal, Lorenzi, *Free-rider Attacks on Model Aggregation in Federated Learning*, AISTATS 2021.
- Zhu et al., *Advanced Free-rider Attacks in Federated Learning*, NeurIPS NFFL Workshop 2021.
