#!/usr/bin/env bash
set -e

# ── 输出实时可见 ──────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ── 项目根目录 ─────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/tmp/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

# ── 1. 安装缺失的依赖 ────────────────────────────────────────────────────────
PIP_PKGS="$PROJECT_DIR/pip_pkgs"
export PYTHONPATH="$PIP_PKGS:$PYTHONPATH"

echo "===== 检查依赖 ====="
MISSING=""
# Import name → pip package name mapping
check_pkg() {
    local import_name="$1"
    local pip_name="$2"
    "$PYTHON_BIN" -c "import $import_name" 2>/dev/null || MISSING="$MISSING $pip_name"
}
check_pkg torch         torch
check_pkg transformers  transformers
check_pkg sklearn       scikit-learn
check_pkg numpy         numpy
check_pkg pandas        pandas
check_pkg matplotlib    matplotlib
check_pkg seaborn       seaborn
check_pkg openpyxl      openpyxl
check_pkg tqdm          tqdm

if [ -n "$MISSING" ]; then
    echo "安装缺失的包:$MISSING"
    export TMPDIR="$PROJECT_DIR/tmp"
    mkdir -p "$TMPDIR"
    "$PYTHON_BIN" -m pip install $MISSING --target "$PIP_PKGS" --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo "依赖安装完成"
else
    echo "依赖已就绪"
fi

# ── 2. 环境变量 ───────────────────────────────────────────────────────────────
export SCIKIT_LEARN_DATA="$PROJECT_DIR/sklearn_data"
mkdir -p "$SCIKIT_LEARN_DATA"

if [ -d "$PROJECT_DIR/model_cache" ]; then
    export MODEL_DIR="$PROJECT_DIR/model_cache"
fi

# ── 3. 运行实验 ───────────────────────────────────────────────────────────────
echo ""
echo "===== 启动实验 ====="
"$PYTHON_BIN" -u main.py "$@"

echo ""
echo "===== 完成，结果保存在 results/ ====="
