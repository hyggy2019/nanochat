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
    --max-seq-len <N>        最大序列长度 (默认: 256)
    --samples-per-update <N> 每次更新的样本数 (默认: 256)
    --batch-size <N>         设备批大小 (默认: 32)
    --optimizer-type <type>  优化器类型: muon 或 rnnps (默认: muon)
    --lr-ratio <R>           学习率缩放比例 [0.0-1.0] (默认: 1.0)
    --embedding-lr <LR>      embedding 基础学习率 (默认: 0.2, 实际 = 基础值 * lr-ratio)
    --unembedding-lr <LR>    unembedding 基础学习率 (默认: 0.004, 实际 = 基础值 * lr-ratio)
    --matrix-lr <LR>         矩阵基础学习率 (默认: 0.02, 实际 = 基础值 * lr-ratio)
    --weight-decay <WD>      权重衰减 (默认: 0.0)
    --rnnps-beta <B>         RNNPS EMA 系数 (默认: 0.95, 仅对 rnnps 优化器有效)
    --rnnps-momentum <M>     RNNPS Nesterov 动量 (默认: 0.9, 仅对 rnnps 优化器有效)
    --row-norm-threshold <T> 行范数阈值 (tau, 默认: 0.0, 仅对 rnnps 优化器有效)
    --norm-scale-variant <V> RNNPS 最大行范数缩放变体 (默认: 0, 仅对 rnnps 优化器有效)
                             0: 标准 RNNPS (无最大行范数缩放)
                             1: 线性缩放 (乘法): scale = default_scale * (1 / max_row_norm)
                             2: 二次方缩放 (乘法): scale = default_scale * (1 / max_row_norm^2)
                             3: 线性替换: scale = 1 / max_row_norm
                             4: 二次方替换: scale = 1 / max_row_norm^2
    --gpus <N>               GPU 数量 (默认: 自动检测)
    --nodes <N>              节点数量 (默认: 1)
    --iterations <N>         训练迭代次数 (默认: -1 自动计算)
    --data-ratio <R>         数据:参数比例 (默认: 20, Chinchilla 最优)
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

    # 使用 LR_RATIO 缩放所有学习率
    ./run_base_train_streaming.sh --lr-ratio=0.5

    # 自定义所有参数，包括 RNNPS 优化器参数
    ./run_base_train_streaming.sh \
        --depth=20 \
        --batch-size=16 \
        --optimizer-type=rnnps \
        --lr-ratio=0.75 \
        --embedding-lr=0.2 \
        --unembedding-lr=0.004 \
        --matrix-lr=0.02 \
        --weight-decay=0.01 \
        --rnnps-beta=0.95 \
        --rnnps-momentum=0.9 \
        --timeout=300 \
        --max-retries=15

    # 调整 RNNPS 参数进行超参数搜索
    ./run_base_train_streaming.sh \
        --optimizer-type=rnnps \
        --matrix-lr=0.008 \
        --rnnps-beta=0.98 \
        --rnnps-momentum=0.95
EOF
}

# ============================================================================
# 默认参数
# ============================================================================

DEPTH=10
MAX_SEQ_LEN=2048
SAMPLES_PER_UPDATE=256
BATCH_SIZE=32

# GPU
NUM_GPUS=4  # 空表示自动检测
NUM_NODES=1
CUDA_VISIBLE_DEVICES=0,1,2,3  # 指定使用哪些 GPU (例如 "0,1,2,3")
NUM_ITERATIONS=-1
TARGET_PARAM_DATA_RATIO=20
STREAMING_TIMEOUT=7200
STREAMING_MAX_RETRIES=10
RUN_NAME=""

# Optimizer
OPTIMIZER_TYPE="rnnps"  # 默认使用 muon，也可以选择 rnnps

# LR Config (基础学习率值)
LR_RATIO=1.0 # [0.0, \inf] 学习率缩放比例
BASE_EMBEDDING_LR=0.2  # Learning rate for embedding parameters (Adam)
BASE_UNEMBEDDING_LR=0.004 # Learning rate for unembedding parameters (Adam)
BASE_MATRIX_LR=0.01 # Learning rate for matrix parameters (Muon/RNNPS)  Muon: 0.02
WEIGHT_DECAY=0.0  # L2 weight decay for embedding/unembedding parameters (Adam)

# RNNPS Optimizer Config
RNNPS_BETA=0.95  # EMA coefficient for RNNPS momentum buffer
RNNPS_MOMENTUM=0.95  # Nesterov coefficient for RNNPS updates
ROW_NORM_THRESHOLD=0.0  # Threshold for row normalization (tau)
NORM_SCALE_VARIANT=1  # Maximum row norm scaling variant (0-4)



# ============================================================================
# 解析命令行参数
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --depth=*)
            DEPTH="${1#*=}"
            shift
            ;;
        --max-seq-len=*)
            MAX_SEQ_LEN="${1#*=}"
            shift
            ;;
        --samples-per-update=*)
            SAMPLES_PER_UPDATE="${1#*=}"
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
        --data-ratio=*)
            TARGET_PARAM_DATA_RATIO="${1#*=}"
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
        --lr-ratio=*)
            LR_RATIO="${1#*=}"
            shift
            ;;
        --embedding-lr=*)
            BASE_EMBEDDING_LR="${1#*=}"
            shift
            ;;
        --unembedding-lr=*)
            BASE_UNEMBEDDING_LR="${1#*=}"
            shift
            ;;
        --weight-decay=*)
            WEIGHT_DECAY="${1#*=}"
            shift
            ;;
        --matrix-lr=*)
            BASE_MATRIX_LR="${1#*=}"
            shift
            ;;
        --rnnps-beta=*)
            RNNPS_BETA="${1#*=}"
            shift
            ;;
        --rnnps-momentum=*)
            RNNPS_MOMENTUM="${1#*=}"
            shift
            ;;
        --row-norm-threshold=*)
            ROW_NORM_THRESHOLD="${1#*=}"
            shift
            ;;
        --norm-scale-variant=*)
            NORM_SCALE_VARIANT="${1#*=}"
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

# ============================================================================
# 计算实际的学习率值 = 基础学习率 * LR_RATIO
# ============================================================================

EMBEDDING_LR=$(awk "BEGIN {printf \"%.6f\", $BASE_EMBEDDING_LR * $LR_RATIO}")
UNEMBEDDING_LR=$(awk "BEGIN {printf \"%.6f\", $BASE_UNEMBEDDING_LR * $LR_RATIO}")
MATRIX_LR=$(awk "BEGIN {printf \"%.6f\", $BASE_MATRIX_LR * $LR_RATIO}")

# 第 152 行附近（在自动检测 GPU 之后）
if [ -z "$RUN_NAME" ]; then
    TIMESTAMP=$(date +%m%d_%H%M)
    if [ "$NUM_ITERATIONS" -eq -1 ]; then
        ITER_TAG="chin${TARGET_PARAM_DATA_RATIO}"
    else
        ITER_TAG="i${NUM_ITERATIONS}"
    fi
    # 格式化学习率为字符串（移除前导零小数点）和 LR_RATIO
    LR_RATIO_TAG=$(echo "$LR_RATIO" | sed 's/^0\./lrratio/' | sed 's/^1\.0$//')
    ELR_TAG=$(echo "$EMBEDDING_LR" | sed 's/^0\./elr/' | sed 's/^0$/elr0/')
    ULR_TAG=$(echo "$UNEMBEDDING_LR" | sed 's/^0\./ulr/' | sed 's/^0$/ulr0/')
    WD_TAG=$(echo "$WEIGHT_DECAY" | sed 's/^0\./wd/' | sed 's/^0$/wd0/')
    MLR_TAG=$(echo "$MATRIX_LR" | sed 's/^0\./mlr/' | sed 's/^0$/mlr0/')
    # 格式化 RNNPS 参数
    BETA_TAG=$(echo "$RNNPS_BETA" | sed 's/^0\./beta/' | sed 's/^0$/beta0/')
    MOMENTUM_TAG=$(echo "$RNNPS_MOMENTUM" | sed 's/^0\./mom/' | sed 's/^0$/mom0/')
    RNORM_TAG=$(echo "$ROW_NORM_THRESHOLD" | sed 's/^0\./rnorm/' | sed 's/^0$/rnorm0/')
    NSV_TAG="nsv${NORM_SCALE_VARIANT}"
    DR_TAG="dr${TARGET_PARAM_DATA_RATIO}"
    SPU_TAG="spu${SAMPLES_PER_UPDATE}"

    if [ -n "$LR_RATIO_TAG" ]; then
        RUN_NAME="depth${DEPTH}_len${MAX_SEQ_LEN}_${OPTIMIZER_TYPE}_b${BATCH_SIZE}_${LR_RATIO_TAG}_${ELR_TAG}_${ULR_TAG}_${WD_TAG}_${MLR_TAG}_${BETA_TAG}_${MOMENTUM_TAG}_${RNORM_TAG}_${NSV_TAG}_${DR_TAG}_${SPU_TAG}_${ITER_TAG}_${TIMESTAMP}"
    else
        RUN_NAME="depth${DEPTH}_len${MAX_SEQ_LEN}_${OPTIMIZER_TYPE}_b${BATCH_SIZE}_${ELR_TAG}_${ULR_TAG}_${WD_TAG}_${MLR_TAG}_${BETA_TAG}_${MOMENTUM_TAG}_${RNORM_TAG}_${NSV_TAG}_${DR_TAG}_${SPU_TAG}_${ITER_TAG}_${TIMESTAMP}"
    fi
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
echo -e "最大序列长度:       ${GREEN}$MAX_SEQ_LEN${NC}"
echo -e "每次更新样本数:     ${GREEN}$SAMPLES_PER_UPDATE${NC}"
echo -e "优化器类型:          ${GREEN}$OPTIMIZER_TYPE${NC}"
echo -e "设备批大小:          ${GREEN}$BATCH_SIZE${NC}"
echo -e "GPU 数量:           ${GREEN}$NUM_GPUS${NC}"
echo -e "节点数量:           ${GREEN}$NUM_NODES${NC}"
echo -e "训练迭代数:         ${GREEN}$NUM_ITERATIONS${NC}"
echo -e "数据:参数比例:       ${GREEN}$TARGET_PARAM_DATA_RATIO${NC}"
echo -e "学习率缩放比例:      ${GREEN}$LR_RATIO${NC}"
echo -e "Embedding 学习率:    ${GREEN}$EMBEDDING_LR (基础: $BASE_EMBEDDING_LR)${NC}"
echo -e "Unembedding 学习率:  ${GREEN}$UNEMBEDDING_LR (基础: $BASE_UNEMBEDDING_LR)${NC}"
echo -e "权重衰减:           ${GREEN}$WEIGHT_DECAY${NC}"
echo -e "矩阵学习率:         ${GREEN}$MATRIX_LR (基础: $BASE_MATRIX_LR)${NC}"
echo -e "RNNPS Beta (EMA):   ${GREEN}$RNNPS_BETA${NC}"
echo -e "RNNPS Momentum:     ${GREEN}$RNNPS_MOMENTUM${NC}"
echo -e "Row Norm Threshold: ${GREEN}$ROW_NORM_THRESHOLD${NC}"
echo -e "Norm Scale Variant: ${GREEN}$NORM_SCALE_VARIANT${NC}"
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
        --max_seq_len=$MAX_SEQ_LEN \
        --samples_per_update=$SAMPLES_PER_UPDATE \
        --device_batch_size=$BATCH_SIZE \
        --num_iterations=$NUM_ITERATIONS \
        --target_param_data_ratio=$TARGET_PARAM_DATA_RATIO \
        --run=$RUN_NAME \
        --optimizer_type=$OPTIMIZER_TYPE \
        --embedding_lr=$EMBEDDING_LR \
        --unembedding_lr=$UNEMBEDDING_LR \
        --weight_decay=$WEIGHT_DECAY \
        --matrix_lr=$MATRIX_LR \
        --rnnps_beta=$RNNPS_BETA \
        --rnnps_momentum=$RNNPS_MOMENTUM \
        --row_norm_threshold=$ROW_NORM_THRESHOLD \
        --norm_scale_variant=$NORM_SCALE_VARIANT \
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
        --max_seq_len=$MAX_SEQ_LEN \
        --samples_per_update=$SAMPLES_PER_UPDATE \
        --device_batch_size=$BATCH_SIZE \
        --num_iterations=$NUM_ITERATIONS \
        --target_param_data_ratio=$TARGET_PARAM_DATA_RATIO \
        --run=$RUN_NAME \
        --optimizer_type=$OPTIMIZER_TYPE \
        --embedding_lr=$EMBEDDING_LR \
        --unembedding_lr=$UNEMBEDDING_LR \
        --weight_decay=$WEIGHT_DECAY \
        --matrix_lr=$MATRIX_LR \
        --rnnps_beta=$RNNPS_BETA \
        --rnnps_momentum=$RNNPS_MOMENTUM \
        --row_norm_threshold=$ROW_NORM_THRESHOLD \
        --norm_scale_variant=$NORM_SCALE_VARIANT \
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
