# 20NEWS-FL — 联想异构智算平台版

**DistilBERT + LoRA 联邦学习 · 搭便车攻击检测 · 每类 Shapley 值分析**

## 平台适配要点

这个版本专为联想异构智算平台的容器环境优化：

| 平台特性 | 适配方式 |
|---|---|
| 非标准 stdout（容器化） | 安全 fallback：`hasattr` 检测 → `write_through` → 双流双写 |
| CUDA 检测可能挂死 | `FORCE_DEVICE` 环境变量可跳过 → 安全降级到 CPU |
| 用户隔离 `I have no name!` | 无影响 — 代码不依赖用户身份 |
| `hf-mirror.com` 不稳定 | 默认直连 HuggingFace（`HF_ENDPOINT=https://huggingface.co`） |
| 首次运行无模型缓存 | 从 HuggingFace 自动下载（需网络） |

## 运行方式

### 方式 1：GPU 模式（默认）

```bash
bash run.sh
```

### 方式 2：强制 CPU 模式（GPU 驱动有问题时）

```bash
FORCE_DEVICE=cpu bash run.sh
```

或者直接：

```bash
FORCE_DEVICE=cpu python3 main.py
```

### 方式 3：快速测试（5 轮，验证流程）

编辑 `main.py` 最后的 `main()` 函数，把参数改成：

```python
base = Config(
    ...
    num_rounds=5,          # 50 → 5
    num_mc_samples=5,      # 30 → 5
    batch_size=16,         # 32 → 16
    ...
)
```

然后 `FORCE_DEVICE=cpu bash run.sh`，约 20 分钟完成。

### 方式 4：如果你已经上传了 model_cache（离线模式）

```bash
# 确保 model_cache/ 和 sklearn_data/ 在项目目录下
ls -d model_cache sklearn_data

# 直接跑
bash run.sh
```

## 诊断输出

正常运行时会看到类似：

```
[20NEWS-FL] Python bootstrap starting
[20NEWS-FL]   Python: 3.12.x
[20NEWS-FL]   Project dir: /storage/main/users/.../school_platform
[20NEWS-FL]   model_cache found: .../school_platform/model_cache
[20NEWS-FL]   HF: direct connect (no mirror)
[20NEWS-FL]   FORCE_DEVICE=auto
[20NEWS-FL] Importing torch...
[20NEWS-FL]   torch 2.x.x imported OK
[20NEWS-FL]   CUDA available: NVIDIA RTX PRO 6000 ...
[20NEWS-FL]   Device: cuda
[20NEWS-FL] Importing project modules...
[20NEWS-FL] All imports OK
[20NEWS-FL] Entering main()
[20NEWS-FL] Config OK — device: cuda
[20NEWS-FL] Loading 20 Newsgroups data...
[data] Fetching 20 Newsgroups...
[data] 20 Newsgroups data fetched successfully
[data] Loading tokenizer: distilbert-base-uncased
[data] Tokenizer loaded
[data] Tokenizing N train texts...
[data] Data ready: train=N, val=N, test=N
[20NEWS-FL] Data loaded — train=N, val=N, test=N
[20NEWS-FL] Starting experiments: baseline, DFR, SDFR, AFR
[20NEWS-FL]   Running: baseline_no_attack
...
```

**如果没有输出**：说明 Python 启动就崩溃了。先跑诊断：

```bash
python3 -c "
import sys
sys.stdout.write('TEST\n'); sys.stdout.flush()
import os, torch
sys.stdout.write('OK torch\n'); sys.stdout.flush()
print('CUDA:', torch.cuda.is_available())
"
```

## 文件结构

```
school_platform/
├── main.py                  # 入口（含安全 CUDA 检测 + [20NEWS-FL] 诊断日志）
├── config.py                # 超参数
├── run.sh                   # 一键运行脚本
├── requirements.txt         # Python 依赖
├── README.md                # 本文件
├── data/newsgroups.py       # 数据加载（含 [data] 进度日志）
├── models/                  # DistilBERT + LoRA
├── fl/                      # FL Server / Client / Aggregation
├── attacks/                 # DFR / SDFR / AFR 攻击
├── contribution/shapley.py  # 蒙特卡洛 Shapley 估计
├── detection/               # Utility Score / 检测器
├── utils/                   # 工具函数
├── visualization/plots.py   # 10 组图表
├── results/                 # 输出目录（运行时生成）
└── sklearn_data/            # 20 Newsgroups 数据缓存（首次运行后生成）
```

## 运行时预估

| GPU | 每轮 | 每实验(50轮) | 4 实验 |
|---|---|---|---|
| RTX PRO 6000 | ~3.5 min | ~3 h | ~12 h |
| CPU only | ~25 min | ~21 h | ~84 h |

## 故障排查

| 症状 | 原因 | 解决 |
|---|---|---|
| `[20NEWS-FL]` 日志都没有 | Python 无法启动 | `python3 --version` 确认 ≥ 3.8 |
| 卡在 `Importing torch...` | PyTorch 安装问题 | `python3 -c "import torch"` 测试 |
| 卡在 `Fetching 20 Newsgroups...` | 无网络，数据未缓存 | 提前上传 `sklearn_data/` 目录 |
| 卡在 `Loading tokenizer...` | 无网络，模型未缓存 | 提前上传 `model_cache/` 目录 |
| CUDA 检测挂死 | GPU 驱动异常 | `FORCE_DEVICE=cpu bash run.sh` |
