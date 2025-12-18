#!/bin/bash

################################################################################
# run_base_train_streaming.sh
#
# 自动寻找可用端口并启动 base_train_streaming.py
# 使用 Hugging Face 在线流式加载（不缓存）
#
# 用法:
#   ./run_base_train_streaming.sh [options]
#
# 选项:
#   --depth <N>              模型深度 (默认: 20)
#   --batch-size <N>         设备批大小 (默认: 32)
#   --gpus <N>               GPU 数量 (默认: 自动检测)
#   --nodes <N>              节点数量 (默认: 1)
#   --iterations <N>         训练迭代次数 (默认: -1 自动计算)
#   --timeout <N>            流式超时时间秒 (默认: 120)
#   --max-retries <N>        流式最大重试次数 (默认: 10)
#   --run-name <name>        Wandb 运行名称 (默认: dummy)
#   --help                   显示帮助信息
#
# 例子:
#   # 默认配置 (自动检测 GPU 数量)
#   ./run_base_train_streaming.sh
#
#   # 指定模型深度和 GPU 数量
#   ./run_base_train_streaming.sh --depth=26 --gpus=8
#
#   # 自定义批大小和网络参数
#   ./run_base_train_streaming.sh --batch-size=16 --timeout=300 --max-retries=15
#
################################################################################

set -e  # 有任何错误立即退出

# ============================================================================
# 颜色输出
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 帮助信息
# ============================================================================

show_help() {
    cat << 'EOF'
用法: ./run_base_train_streaming.sh [options]

自动寻找可用端口并启动 base_train_streaming.py with streaming mode

选项:
    --depth <N>              模型深度 (默认: 20)
    --batch-size <N>         设备批大小 (默认: 32)
    --optimizer-type <type>  优化器类型: muon 或 rnnps (默认: muon)
    --gpus <N>               GPU 数量 (默认: 自动检测)
    --nodes <N>              节点数量 (默认: 1)
    --iterations <N>         训练迭代次数 (默认: -1 自动计算)
    --timeout <N>            流式超时时间秒 (默认: 7200)
    --max-retries <N>        流式最大重试次数 (默认: 10)
    --run-name <name>        Wandb 运行名称 (默认: 自动生成)
    --help                   显示此帮助信息

例子:
    # 默认配置 (使用 muon 优化器)
    ./run_base_train_streaming.sh

    # 使用 RNNPS 优化器
    ./run_base_train_streaming.sh --optimizer-type=rnnps

    # 指定模型深度和优化器
    ./run_base_train_streaming.sh --depth=26 --optimizer-type=rnnps

    # 自定义所有参数
    ./run_base_train_streaming.sh \
        --depth=20 \
        --batch-size=16 \
        --optimizer-type=rnnps \
        --timeout=300 \
        --max-retries=15
EOF
}

# ============================================================================
# 默认参数
# ============================================================================

DEPTH=20
BATCH_SIZE=32
NUM_GPUS=2  # 空表示自动检测
NUM_NODES=1
NUM_ITERATIONS=-1
STREAMING_TIMEOUT=7200
STREAMING_MAX_RETRIES=10
RUN_NAME=""
OPTIMIZER_TYPE="rnnps"  # 默认使用 muon，也可以选择 rnnps
WEIGHT_DECAY=0.0  # L2 weight decay for matrix parameters
CUDA_VISIBLE_DEVICES=2,5  # 指定使用哪些 GPU (例如 "0,1,2,3")

# ============================================================================
# 解析命令行参数
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --depth=*)
            DEPTH="${1#*=}"
            shift
            ;;
        --batch-size=*)
            BATCH_SIZE="${1#*=}"
            shift
            ;;
        --gpus=*)
            NUM_GPUS="${1#*=}"
            shift
            ;;
        --nodes=*)
            NUM_NODES="${1#*=}"
            shift
            ;;
        --iterations=*)
            NUM_ITERATIONS="${1#*=}"
            shift
            ;;
        --timeout=*)
            STREAMING_TIMEOUT="${1#*=}"
            shift
            ;;
        --max-retries=*)
            STREAMING_MAX_RETRIES="${1#*=}"
            shift
            ;;
        --run-name=*)
            RUN_NAME="${1#*=}"
            shift
            ;;
        --optimizer-type=*)
            OPTIMIZER_TYPE="${1#*=}"
            shift
            ;;
        --weight-decay=*)
            WEIGHT_DECAY="${1#*=}"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 第 152 行附近（在自动检测 GPU 之后）
if [ -z "$RUN_NAME" ]; then
    TIMESTAMP=$(date +%m%d_%H%M)
    if [ "$NUM_ITERATIONS" -eq -1 ]; then
        ITER_TAG="chin20"  # Chinchilla ratio 20
    else
        ITER_TAG="i${NUM_ITERATIONS}"
    fi
    # 格式化weight_decay为字符串（移除前导零小数点）
    WD_TAG=$(echo "$WEIGHT_DECAY" | sed 's/^0\./wd/' | sed 's/^0$/wd0/')
    RUN_NAME="d${DEPTH}_${OPTIMIZER_TYPE}_b${BATCH_SIZE}_${WD_TAG}_${ITER_TAG}_${TIMESTAMP}"
    echo -e "${YELLOW}⚠ 自动生成 Wandb run_name: ${GREEN}$RUN_NAME${NC}"
fi

# ============================================================================
# 自动检测 GPU 数量
# ============================================================================

if [ -z "$NUM_GPUS" ]; then
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
        echo -e "${GREEN}✓ 自动检测到 GPU 数量: $NUM_GPUS${NC}"
    else
        echo -e "${YELLOW}⚠ 无法检测 GPU（nvidia-smi 不可用），使用默认值 1${NC}"
        NUM_GPUS=1
    fi
else
    echo -e "${GREEN}✓ 使用指定的 GPU 数量: $NUM_GPUS${NC}"
fi

# ============================================================================
# 寻找可用的端口
# ============================================================================

find_available_port() {
    local port=29500
    local max_port=29600

    while [ $port -le $max_port ]; do
        # 检查端口是否被占用
        if ! netstat -tuln 2>/dev/null | grep -q ":$port " && \
           ! ss -tuln 2>/dev/null | grep -q ":$port "; then
            echo $port
            return 0
        fi
        port=$((port + 1))
    done

    # 如果都占用了，使用随机端口
    echo $((29500 + RANDOM % 100))
}

MASTER_PORT=$(find_available_port)
echo -e "${GREEN}✓ 使用端口: $MASTER_PORT${NC}"

# ============================================================================
# 显示配置信息
# ============================================================================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}📋 训练配置${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "模型深度:           ${GREEN}$DEPTH${NC}"
echo -e "优化器类型:          ${GREEN}$OPTIMIZER_TYPE${NC}"
echo -e "设备批大小:          ${GREEN}$BATCH_SIZE${NC}"
echo -e "GPU 数量:           ${GREEN}$NUM_GPUS${NC}"
echo -e "节点数量:           ${GREEN}$NUM_NODES${NC}"
echo -e "训练迭代数:         ${GREEN}$NUM_ITERATIONS${NC}"
echo -e "权重衰减:           ${GREEN}$WEIGHT_DECAY${NC}"
echo -e "Wandb 运行名:       ${GREEN}$RUN_NAME${NC}"
echo ""
echo -e "${BLUE}📡 流式加载配置${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "使用流式加载:       ${GREEN}true (不缓存)${NC}"
echo -e "超时时间:           ${GREEN}${STREAMING_TIMEOUT}s${NC}"
echo -e "最大重试次数:       ${GREEN}$STREAMING_MAX_RETRIES${NC}"
echo ""
echo -e "${BLUE}🌐 分布式训练配置${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "Master 端口:        ${GREEN}$MASTER_PORT${NC}"
echo -e "Master IP:          ${GREEN}127.0.0.1${NC}"
echo ""

# ============================================================================
# 环境变量设置
# ============================================================================

# 设置缓存目录到 /scratch（有足够空间）
export NANOCHAT_BASE_DIR="/scratch/nanochat_cache"
export HF_HOME="/scratch/nanochat_cache/huggingface"

# 创建缓存目录
mkdir -p /scratch/nanochat_cache/huggingface
mkdir -p /scratch/nanochat_cache/tokenizer
mkdir -p /scratch/nanochat_cache/base_data
mkdir -p /scratch/nanochat_cache/base_checkpoints

# 设置可见的 GPU（如果指定）
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    echo -e "${GREEN}✓ 设置 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES${NC}"
    # 重新计算 GPU 数量
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    echo -e "${YELLOW}⚠ GPU 数量已更新为: $NUM_GPUS${NC}"
fi

echo -e "${BLUE}🔗 设置缓存目录链接${NC}"
DEFAULT_CACHE="$HOME/.cache/nanochat"
if [ ! -d "$DEFAULT_CACHE" ]; then
    mkdir -p "$(dirname "$DEFAULT_CACHE")"
    ln -s /scratch/nanochat_cache "$DEFAULT_CACHE"
    echo -e "${GREEN}✓ 创建符号链接: $DEFAULT_CACHE -> /scratch/nanochat_cache${NC}"
else
    echo -e "${YELLOW}⚠ 目录已存在: $DEFAULT_CACHE${NC}"
fi
echo ""

# OMP 线程数（多 GPU 训练推荐设置为 1）
export OMP_NUM_THREADS=1

# 可扩展显存配置（用于防止 OOM）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 禁用 TF32（如果需要更高精度）
# export NVIDIA_TF32_OVERRIDE=0

# 显示缓存配置
echo ""
echo -e "${BLUE}💾 缓存配置${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "缓存根目录:         ${GREEN}$NANOCHAT_CACHE_DIR${NC}"
echo -e "HF 缓存:            ${GREEN}$HF_HOME${NC}"
echo -e "Tokenizer 缓存:     ${GREEN}/scratch/nanochat_cache/tokenizer${NC}"
echo ""

echo -e "${BLUE}🚀 启动训练...${NC}"
echo ""

# ============================================================================
# 构建 torchrun 命令
# ============================================================================

if [ "$NUM_NODES" -eq 1 ]; then
    # 单节点训练
    torchrun \
        --standalone \
        --nproc_per_node=$NUM_GPUS \
        -m scripts.base_train_streaming \
        -- \
        --depth=$DEPTH \
        --device_batch_size=$BATCH_SIZE \
        --num_iterations=$NUM_ITERATIONS \
        --run=$RUN_NAME \
        --optimizer_type=$OPTIMIZER_TYPE \
        --weight_decay=$WEIGHT_DECAY \
        --use_streaming=True \
        --cache_streaming=False \
        --streaming_timeout=$STREAMING_TIMEOUT \
        --streaming_max_retries=$STREAMING_MAX_RETRIES
else
    # 多节点训练（需要设置 MASTER_ADDR 和 MASTER_PORT）
    export MASTER_ADDR="127.0.0.1"
    export MASTER_PORT=$MASTER_PORT

    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=$NUM_NODES \
        --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        -m scripts.base_train_streaming \
        -- \
        --depth=$DEPTH \
        --device_batch_size=$BATCH_SIZE \
        --num_iterations=$NUM_ITERATIONS \
        --run=$RUN_NAME \
        --optimizer_type=$OPTIMIZER_TYPE \
        --weight_decay=$WEIGHT_DECAY \
        --use_streaming=True \
        --cache_streaming=False \
        --streaming_timeout=$STREAMING_TIMEOUT \
        --streaming_max_retries=$STREAMING_MAX_RETRIES
fi

# 如果训练成功，显示完成信息
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✅ 训练完成！${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
else
    echo ""
    echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}❌ 训练失败！${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
    exit 1
fi
