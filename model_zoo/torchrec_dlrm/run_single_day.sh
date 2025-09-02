#!/bin/bash

# Usage:
# (with nohup/screen)
# ./run_single_day.sh [PROCESSED_DATASET_PATH] [BATCH_SIZE] [LEARNING_RATE] [EPOCHS] [DATASET_SIZE]

DEFAULT_PROCESSED_DATASET_PATH="./processed_day_0_data"
DEFAULT_BATCH_SIZE=1024
DEFAULT_LEARNING_RATE=0.005
DEFAULT_EPOCHS=1
DEFAULT_DATASET_SIZE=4194304

DATASET_SIZE=${1:-$DEFAULT_DATASET_SIZE}
PROCESSED_DATASET_PATH=${2:-$DEFAULT_PROCESSED_DATASET_PATH}
BATCH_SIZE=${3:-$DEFAULT_BATCH_SIZE}
LEARNING_RATE=${4:-$DEFAULT_LEARNING_RATE}
EPOCHS=${5:-$DEFAULT_EPOCHS}

DLRM_PATH="$(pwd)"
SINGLE_DAY_TEST="${DLRM_PATH}/tests/dlrm_main_single_day.py"
source ${DLRM_PATH}/dlrm_venv/bin/activate

echo "=== TorchRec DLRM Single Day Training ==="
echo "Dataset path: $PROCESSED_DATASET_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Dataset Size (for log naming): $DATASET_SIZE"
echo ""

if [ ! -d "$PROCESSED_DATASET_PATH" ]; then
    echo "Error: Processed dataset not found at $PROCESSED_DATASET_PATH"
    echo "Please run the preprocessing script first:"
    echo "bash scripts/process_single_day.sh <raw_data_dir> $PROCESSED_DATASET_PATH"
    exit 1
fi

required_files=("day_0_dense.npy" "day_0_sparse.npy" "day_0_labels.npy")
for file in "${required_files[@]}"; do
    if [ ! -f "$PROCESSED_DATASET_PATH/$file" ]; then
        echo "Error: Required file $file not found in $PROCESSED_DATASET_PATH"
        exit 1
    fi
done

echo "✓ All required data files found"
echo ""

echo "Starting training..."
torchrun --nnodes 1 \
    --nproc_per_node 1 \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost \
    --rdzv_id 54321 \
    --role trainer $SINGLE_DAY_TEST \
    --single_day_mode \
    --in_memory_binary_criteo_path $PROCESSED_DATASET_PATH \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --pin_memory \
    --mmap_mode \
    --embedding_dim 128 \
    --adagrad > training_output.$DATASET_SIZE.$(date +%Y%m%d%H%M%S).log 2>&1

echo ""
echo "Training completed!"
