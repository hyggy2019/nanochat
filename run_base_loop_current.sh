#!/bin/bash

################################################################################
# run_loop.sh
#
# 支持多参数循环的训练脚本
# 使用笛卡尔积遍历所有参数组合，失败时继续运行后续实验
#
# 用法:
#   ./run_loop.sh [options]
#
# 参数支持逗号分隔的列表，例如:
#   ./run_loop.sh --depth=10,20,26 --norm-scale-variant=0,1,2
#   将运行 3*3=9 个实验
#
################################################################################

# 不使用 set -e，因为需要在失败时继续运行

# ============================================================================
# 颜色输出
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# 帮助信息
# ============================================================================

show_help() {
    cat << 'EOF'
用法: ./run_loop.sh [options]

支持多参数循环的训练脚本，使用笛卡尔积遍历所有参数组合。
参数可以使用逗号分隔指定多个值。

可循环参数 (支持逗号分隔多值):
    --depth=<N1,N2,...>              模型深度 (默认: 10)
    --optimizer-type=<t1,t2,...>     优化器类型: muon 或 rnnps (默认: rnnps)
    --lr-ratio=<R1,R2,...>           学习率缩放比例 (默认: 1.0)
    --embedding-lr=<LR1,LR2,...>     embedding 基础学习率 (默认: 0.2)
    --unembedding-lr=<LR1,LR2,...>   unembedding 基础学习率 (默认: 0.004)
    --matrix-lr=<LR1,LR2,...>        矩阵基础学习率 (默认: 0.01)
    --weight-decay=<WD1,WD2,...>     权重衰减 (默认: 0.0)
    --rnnps-beta=<B1,B2,...>         RNNPS EMA 系数 (默认: 0.95)
    --rnnps-momentum=<M1,M2,...>     RNNPS Nesterov 动量 (默认: 0.90)
    --norm-scale-variant=<V1,V2,...> RNNPS 最大行范数缩放变体 0-4 (默认: 0)
    --data-ratio=<R1,R2,...>         数据:参数比例 (默认: 20, Chinchilla 最优)
    --samples-per-update=<N1,N2,...> 每次更新的样本数 (默认: 256)

固定参数 (单值):
    --max-seq-len=<N>                最大序列长度 (默认: 2048)
    --batch-size=<N>                 设备批大小 (默认: 32)
    --gpus=<N>                       GPU 数量 (默认: 4)
    --cuda-devices=<0,1,2,3>         指定 CUDA 设备 (默认: 0,1,2,3)
    --nodes=<N>                      节点数量 (默认: 1)
    --iterations=<N>                 训练迭代次数 (默认: -1 自动计算)
    --timeout=<N>                    流式超时时间秒 (默认: 7200)
    --max-retries=<N>                流式最大重试次数 (默认: 10)
    --help                           显示此帮助信息

例子:
    # 单个实验 (默认配置)
    ./run_loop.sh

    # 测试多个深度
    ./run_loop.sh --depth=10,20,26

    # 测试多个参数组合 (笛卡尔积: 2*3=6 个实验)
    ./run_loop.sh --depth=10,20 --norm-scale-variant=0,1,2

    # 超参数搜索
    ./run_loop.sh \
        --optimizer-type=rnnps \
        --matrix-lr=0.005,0.01,0.02 \
        --norm-scale-variant=0,1,2,3,4
EOF
}

# ============================================================================
# 默认参数
# ============================================================================

# 可循环参数 (数组形式)
DEPTH_LIST=(10 15)
OPTIMIZER_TYPE_LIST=("muon")
LR_RATIO_LIST=(1.0)
BASE_EMBEDDING_LR_LIST=(0.2)
BASE_UNEMBEDDING_LR_LIST=(0.004)
BASE_MATRIX_LR_LIST=(0.02)
WEIGHT_DECAY_LIST=(0.0)
RNNPS_BETA_LIST=(0.95)
RNNPS_MOMENTUM_LIST=(0.95)
NORM_SCALE_VARIANT_LIST=(0)
TARGET_PARAM_DATA_RATIO_LIST=(50)
SAMPLES_PER_UPDATE_LIST=(256 512)

# 固定参数 (单值)
MAX_SEQ_LEN=2048
BATCH_SIZE=32
NUM_GPUS=4
NUM_NODES=1
CUDA_VISIBLE_DEVICES="0,1,2,3"
NUM_ITERATIONS=-1
STREAMING_TIMEOUT=7200
STREAMING_MAX_RETRIES=10

# ============================================================================
# 解析命令行参数 (支持逗号分隔的列表)
# ============================================================================

# 辅助函数：将逗号分隔的字符串转换为数组
parse_list() {
    local input="$1"
    local -n arr=$2
    IFS=',' read -ra arr <<< "$input"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        # 可循环参数
        --depth=*)
            parse_list "${1#*=}" DEPTH_LIST
            shift
            ;;
        --optimizer-type=*)
            parse_list "${1#*=}" OPTIMIZER_TYPE_LIST
            shift
            ;;
        --lr-ratio=*)
            parse_list "${1#*=}" LR_RATIO_LIST
            shift
            ;;
        --embedding-lr=*)
            parse_list "${1#*=}" BASE_EMBEDDING_LR_LIST
            shift
            ;;
        --unembedding-lr=*)
            parse_list "${1#*=}" BASE_UNEMBEDDING_LR_LIST
            shift
            ;;
        --matrix-lr=*)
            parse_list "${1#*=}" BASE_MATRIX_LR_LIST
            shift
            ;;
        --weight-decay=*)
            parse_list "${1#*=}" WEIGHT_DECAY_LIST
            shift
            ;;
        --rnnps-beta=*)
            parse_list "${1#*=}" RNNPS_BETA_LIST
            shift
            ;;
        --rnnps-momentum=*)
            parse_list "${1#*=}" RNNPS_MOMENTUM_LIST
            shift
            ;;
        --norm-scale-variant=*)
            parse_list "${1#*=}" NORM_SCALE_VARIANT_LIST
            shift
            ;;
        --data-ratio=*)
            parse_list "${1#*=}" TARGET_PARAM_DATA_RATIO_LIST
            shift
            ;;
        --samples-per-update=*)
            parse_list "${1#*=}" SAMPLES_PER_UPDATE_LIST
            shift
            ;;
        # 固定参数
        --max-seq-len=*)
            MAX_SEQ_LEN="${1#*=}"
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
        --cuda-devices=*)
            CUDA_VISIBLE_DEVICES="${1#*=}"
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
# 计算总实验数量
# ============================================================================

TOTAL_EXPERIMENTS=$((
    ${#DEPTH_LIST[@]} *
    ${#OPTIMIZER_TYPE_LIST[@]} *
    ${#LR_RATIO_LIST[@]} *
    ${#BASE_EMBEDDING_LR_LIST[@]} *
    ${#BASE_UNEMBEDDING_LR_LIST[@]} *
    ${#BASE_MATRIX_LR_LIST[@]} *
    ${#WEIGHT_DECAY_LIST[@]} *
    ${#RNNPS_BETA_LIST[@]} *
    ${#RNNPS_MOMENTUM_LIST[@]} *
    ${#NORM_SCALE_VARIANT_LIST[@]} *
    ${#TARGET_PARAM_DATA_RATIO_LIST[@]} *
    ${#SAMPLES_PER_UPDATE_LIST[@]}
))

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}🔄 多参数循环实验配置${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}可循环参数:${NC}"
echo -e "  DEPTH:              ${GREEN}${DEPTH_LIST[*]}${NC} (${#DEPTH_LIST[@]} 个值)"
echo -e "  OPTIMIZER_TYPE:     ${GREEN}${OPTIMIZER_TYPE_LIST[*]}${NC} (${#OPTIMIZER_TYPE_LIST[@]} 个值)"
echo -e "  LR_RATIO:           ${GREEN}${LR_RATIO_LIST[*]}${NC} (${#LR_RATIO_LIST[@]} 个值)"
echo -e "  BASE_EMBEDDING_LR:  ${GREEN}${BASE_EMBEDDING_LR_LIST[*]}${NC} (${#BASE_EMBEDDING_LR_LIST[@]} 个值)"
echo -e "  BASE_UNEMBEDDING_LR:${GREEN}${BASE_UNEMBEDDING_LR_LIST[*]}${NC} (${#BASE_UNEMBEDDING_LR_LIST[@]} 个值)"
echo -e "  BASE_MATRIX_LR:     ${GREEN}${BASE_MATRIX_LR_LIST[*]}${NC} (${#BASE_MATRIX_LR_LIST[@]} 个值)"
echo -e "  WEIGHT_DECAY:       ${GREEN}${WEIGHT_DECAY_LIST[*]}${NC} (${#WEIGHT_DECAY_LIST[@]} 个值)"
echo -e "  RNNPS_BETA:         ${GREEN}${RNNPS_BETA_LIST[*]}${NC} (${#RNNPS_BETA_LIST[@]} 个值)"
echo -e "  RNNPS_MOMENTUM:     ${GREEN}${RNNPS_MOMENTUM_LIST[*]}${NC} (${#RNNPS_MOMENTUM_LIST[@]} 个值)"
echo -e "  NORM_SCALE_VARIANT: ${GREEN}${NORM_SCALE_VARIANT_LIST[*]}${NC} (${#NORM_SCALE_VARIANT_LIST[@]} 个值)"
echo -e "  DATA_RATIO:         ${GREEN}${TARGET_PARAM_DATA_RATIO_LIST[*]}${NC} (${#TARGET_PARAM_DATA_RATIO_LIST[@]} 个值)"
echo -e "  SAMPLES_PER_UPDATE: ${GREEN}${SAMPLES_PER_UPDATE_LIST[*]}${NC} (${#SAMPLES_PER_UPDATE_LIST[@]} 个值)"
echo ""
echo -e "${CYAN}固定参数:${NC}"
echo -e "  MAX_SEQ_LEN:        ${GREEN}${MAX_SEQ_LEN}${NC}"
echo -e "  BATCH_SIZE:         ${GREEN}${BATCH_SIZE}${NC}"
echo ""
echo -e "${CYAN}总实验数量: ${YELLOW}${TOTAL_EXPERIMENTS}${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# ============================================================================
# 环境变量设置 (只执行一次)
# ============================================================================

# 设置缓存目录到 ~/Nanochat（有足够空间）
CACHE_DIR="$HOME/Nanochat"
export NANOCHAT_BASE_DIR="$CACHE_DIR"
export HF_HOME="$CACHE_DIR/huggingface"

# 创建缓存目录
mkdir -p "$CACHE_DIR/huggingface"
mkdir -p "$CACHE_DIR/tokenizer"
mkdir -p "$CACHE_DIR/base_data"
mkdir -p "$CACHE_DIR/base_checkpoints"

# 设置可见的 GPU
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo -e "${GREEN}✓ 设置 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES${NC}"

# 重新计算 GPU 数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo -e "${GREEN}✓ GPU 数量: $NUM_GPUS${NC}"

# 设置缓存目录链接
DEFAULT_CACHE="$HOME/.cache/nanochat"
if [ ! -e "$DEFAULT_CACHE" ]; then
    mkdir -p "$(dirname "$DEFAULT_CACHE")"
    ln -s "$CACHE_DIR" "$DEFAULT_CACHE" 2>/dev/null || true
fi

# OMP 线程数（多 GPU 训练推荐设置为 1）
export OMP_NUM_THREADS=1

# 可扩展显存配置（用于防止 OOM）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# 寻找可用端口的函数
# ============================================================================

find_available_port() {
    local port=29500
    local max_port=29600

    while [ $port -le $max_port ]; do
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

# ============================================================================
# 单个实验执行函数
# ============================================================================

run_single_experiment() {
    local DEPTH=$1
    local OPTIMIZER_TYPE=$2
    local LR_RATIO=$3
    local BASE_EMBEDDING_LR=$4
    local BASE_UNEMBEDDING_LR=$5
    local BASE_MATRIX_LR=$6
    local WEIGHT_DECAY=$7
    local RNNPS_BETA=$8
    local RNNPS_MOMENTUM=$9
    local NORM_SCALE_VARIANT=${10}
    local TARGET_PARAM_DATA_RATIO=${11}
    local SAMPLES_PER_UPDATE=${12}

    # 计算实际学习率
    local EMBEDDING_LR=$(awk "BEGIN {printf \"%.6f\", $BASE_EMBEDDING_LR * $LR_RATIO}")
    local UNEMBEDDING_LR=$(awk "BEGIN {printf \"%.6f\", $BASE_UNEMBEDDING_LR * $LR_RATIO}")
    local MATRIX_LR=$(awk "BEGIN {printf \"%.6f\", $BASE_MATRIX_LR * $LR_RATIO}")

    # 生成 RUN_NAME
    local TIMESTAMP=$(date +%m%d_%H%M%S)
    local ITER_TAG
    if [ "$NUM_ITERATIONS" -eq -1 ]; then
        ITER_TAG="chin${TARGET_PARAM_DATA_RATIO}"
    else
        ITER_TAG="i${NUM_ITERATIONS}"
    fi

    local LR_RATIO_TAG=$(echo "$LR_RATIO" | sed 's/^0\./lrratio/' | sed 's/^1\.0$//')
    local ELR_TAG=$(echo "$EMBEDDING_LR" | sed 's/^0\./elr/' | sed 's/^0$/elr0/')
    local ULR_TAG=$(echo "$UNEMBEDDING_LR" | sed 's/^0\./ulr/' | sed 's/^0$/ulr0/')
    local WD_TAG=$(echo "$WEIGHT_DECAY" | sed 's/^0\./wd/' | sed 's/^0$/wd0/')
    local MLR_TAG=$(echo "$MATRIX_LR" | sed 's/^0\./mlr/' | sed 's/^0$/mlr0/')
    local BETA_TAG=$(echo "$RNNPS_BETA" | sed 's/^0\./beta/' | sed 's/^0$/beta0/')
    local MOMENTUM_TAG=$(echo "$RNNPS_MOMENTUM" | sed 's/^0\./mom/' | sed 's/^0$/mom0/')
    local NSV_TAG="nsv${NORM_SCALE_VARIANT}"
    local DR_TAG="dr${TARGET_PARAM_DATA_RATIO}"
    local SPU_TAG="spu${SAMPLES_PER_UPDATE}"

    local RUN_NAME
    if [ -n "$LR_RATIO_TAG" ]; then
        RUN_NAME="depth${DEPTH}_len${MAX_SEQ_LEN}_${OPTIMIZER_TYPE}_b${BATCH_SIZE}_${LR_RATIO_TAG}_${ELR_TAG}_${ULR_TAG}_${WD_TAG}_${MLR_TAG}_${BETA_TAG}_${MOMENTUM_TAG}_${NSV_TAG}_${DR_TAG}_${SPU_TAG}_${ITER_TAG}_${TIMESTAMP}"
    else
        RUN_NAME="depth${DEPTH}_len${MAX_SEQ_LEN}_${OPTIMIZER_TYPE}_b${BATCH_SIZE}_${ELR_TAG}_${ULR_TAG}_${WD_TAG}_${MLR_TAG}_${BETA_TAG}_${MOMENTUM_TAG}_${NSV_TAG}_${DR_TAG}_${SPU_TAG}_${ITER_TAG}_${TIMESTAMP}"
    fi

    # 寻找可用端口
    local MASTER_PORT=$(find_available_port)

    # 显示配置
    echo ""
    echo -e "${BLUE}───────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}📋 实验配置${NC}"
    echo -e "${BLUE}───────────────────────────────────────────────────────────${NC}"
    echo -e "模型深度:           ${GREEN}$DEPTH${NC}"
    echo -e "优化器类型:         ${GREEN}$OPTIMIZER_TYPE${NC}"
    echo -e "学习率缩放:         ${GREEN}$LR_RATIO${NC}"
    echo -e "Embedding LR:       ${GREEN}$EMBEDDING_LR${NC}"
    echo -e "Unembedding LR:     ${GREEN}$UNEMBEDDING_LR${NC}"
    echo -e "Matrix LR:          ${GREEN}$MATRIX_LR${NC}"
    echo -e "Weight Decay:       ${GREEN}$WEIGHT_DECAY${NC}"
    echo -e "RNNPS Beta:         ${GREEN}$RNNPS_BETA${NC}"
    echo -e "RNNPS Momentum:     ${GREEN}$RNNPS_MOMENTUM${NC}"
    echo -e "Norm Scale Variant: ${GREEN}$NORM_SCALE_VARIANT${NC}"
    echo -e "Data Ratio:         ${GREEN}$TARGET_PARAM_DATA_RATIO${NC}"
    echo -e "Samples Per Update: ${GREEN}$SAMPLES_PER_UPDATE${NC}"
    echo -e "Master Port:        ${GREEN}$MASTER_PORT${NC}"
    echo -e "Run Name:           ${GREEN}$RUN_NAME${NC}"
    echo ""

    # 执行训练
    if [ "$NUM_NODES" -eq 1 ]; then
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
            --norm_scale_variant=$NORM_SCALE_VARIANT \
            --use_streaming=True \
            --cache_streaming=False \
            --streaming_timeout=$STREAMING_TIMEOUT \
            --streaming_max_retries=$STREAMING_MAX_RETRIES
    else
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
            --norm_scale_variant=$NORM_SCALE_VARIANT \
            --use_streaming=True \
            --cache_streaming=False \
            --streaming_timeout=$STREAMING_TIMEOUT \
            --streaming_max_retries=$STREAMING_MAX_RETRIES
    fi

    return $?
}

# ============================================================================
# 主循环：笛卡尔积遍历所有参数组合
# ============================================================================

CURRENT_EXP=0
SUCCESS_COUNT=0
FAILED_EXPS=()

echo -e "${BLUE}🚀 开始实验循环...${NC}"
echo ""

for DEPTH in "${DEPTH_LIST[@]}"; do
for OPTIMIZER_TYPE in "${OPTIMIZER_TYPE_LIST[@]}"; do
for LR_RATIO in "${LR_RATIO_LIST[@]}"; do
for BASE_EMBEDDING_LR in "${BASE_EMBEDDING_LR_LIST[@]}"; do
for BASE_UNEMBEDDING_LR in "${BASE_UNEMBEDDING_LR_LIST[@]}"; do
for BASE_MATRIX_LR in "${BASE_MATRIX_LR_LIST[@]}"; do
for WEIGHT_DECAY in "${WEIGHT_DECAY_LIST[@]}"; do
for RNNPS_BETA in "${RNNPS_BETA_LIST[@]}"; do
for RNNPS_MOMENTUM in "${RNNPS_MOMENTUM_LIST[@]}"; do
for NORM_SCALE_VARIANT in "${NORM_SCALE_VARIANT_LIST[@]}"; do
for TARGET_PARAM_DATA_RATIO in "${TARGET_PARAM_DATA_RATIO_LIST[@]}"; do
for SAMPLES_PER_UPDATE in "${SAMPLES_PER_UPDATE_LIST[@]}"; do

    CURRENT_EXP=$((CURRENT_EXP + 1))

    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}🧪 实验 ${CURRENT_EXP}/${TOTAL_EXPERIMENTS}${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"

    # 运行实验
    if run_single_experiment \
        "$DEPTH" \
        "$OPTIMIZER_TYPE" \
        "$LR_RATIO" \
        "$BASE_EMBEDDING_LR" \
        "$BASE_UNEMBEDDING_LR" \
        "$BASE_MATRIX_LR" \
        "$WEIGHT_DECAY" \
        "$RNNPS_BETA" \
        "$RNNPS_MOMENTUM" \
        "$NORM_SCALE_VARIANT" \
        "$TARGET_PARAM_DATA_RATIO" \
        "$SAMPLES_PER_UPDATE"
    then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo -e "${GREEN}✅ 实验 ${CURRENT_EXP}/${TOTAL_EXPERIMENTS} 成功${NC}"
    else
        FAILED_EXPS+=("${CURRENT_EXP}: D${DEPTH}_${OPTIMIZER_TYPE}_LR${LR_RATIO}_MLR${BASE_MATRIX_LR}_WD${WEIGHT_DECAY}_B${RNNPS_BETA}_M${RNNPS_MOMENTUM}_NSV${NORM_SCALE_VARIANT}_DR${TARGET_PARAM_DATA_RATIO}_SPU${SAMPLES_PER_UPDATE}")
        echo -e "${RED}❌ 实验 ${CURRENT_EXP}/${TOTAL_EXPERIMENTS} 失败${NC}"
    fi

done
done
done
done
done
done
done
done
done
done
done
done

# ============================================================================
# 汇总结果
# ============================================================================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}📊 实验汇总${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "总实验数:   ${CYAN}${TOTAL_EXPERIMENTS}${NC}"
echo -e "成功:       ${GREEN}${SUCCESS_COUNT}${NC}"
echo -e "失败:       ${RED}${#FAILED_EXPS[@]}${NC}"
echo ""

if [ ${#FAILED_EXPS[@]} -gt 0 ]; then
    echo -e "${RED}失败的实验:${NC}"
    for exp in "${FAILED_EXPS[@]}"; do
        echo -e "  ${RED}• ${exp}${NC}"
    done
    echo ""
fi

if [ ${#FAILED_EXPS[@]} -eq 0 ]; then
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✅ 所有实验完成！${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    exit 0
else
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}⚠ 部分实验失败，请检查日志${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
    exit 1
fi
