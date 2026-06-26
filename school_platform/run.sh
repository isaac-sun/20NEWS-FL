#!/usr/bin/env bash
# =============================================================================
# 20NEWS-FL — 联想异构智算平台 运行脚本
# =============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "===== 20NEWS-FL 平台诊断 ====="
echo "项目目录: $PROJECT_DIR"
echo "Python:   $(python3 --version 2>&1 || echo 'NOT FOUND')"

# ═════════════════════════════════════════════════════════════════════════════
# 1. 环境变量
# ═════════════════════════════════════════════════════════════════════════════
export PYTHONUNBUFFERED=1

# sklearn 数据缓存
export SCIKIT_LEARN_DATA="$PROJECT_DIR/sklearn_data"
mkdir -p "$SCIKIT_LEARN_DATA"
echo "sklearn_data: $SCIKIT_LEARN_DATA"

# 模型缓存（本地优先）
if [ -d "$PROJECT_DIR/model_cache" ]; then
    export MODEL_DIR="$PROJECT_DIR/model_cache"
    echo "model_cache:  $MODEL_DIR (已存在)"
else
    echo "model_cache:  未找到（将从 HuggingFace 下载）"
fi

# ═════════════════════════════════════════════════════════════════════════════
# 2. 设备选择
# ═════════════════════════════════════════════════════════════════════════════
# 如果 GPU 驱动有问题，设 FORCE_DEVICE=cpu 跳过 CUDA 检测
#   export FORCE_DEVICE=cpu
# 如果 GPU 正常，不设，代码自动检测
echo "FORCE_DEVICE: ${FORCE_DEVICE:-auto}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-(all)}"

# ═════════════════════════════════════════════════════════════════════════════
# 3. 运行
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "===== 启动实验 ====="
python3 main.py

echo ""
echo "===== 完成，结果保存在 results/ ====="
