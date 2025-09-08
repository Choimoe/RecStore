#!/bin/bash

# Usage:
# (with nohup/screen)
# ./run_single_day.sh --help
# Example:
# ./run_single_day.sh --custom --dataset-size 65535

DEFAULT_PROCESSED_DATASET_PATH="./processed_day_0_data"
DEFAULT_BATCH_SIZE=1024
DEFAULT_LEARNING_RATE=0.005
DEFAULT_EPOCHS=1
DEFAULT_DATASET_SIZE=4194304

DLRM_PATH="$(pwd)"
VENV_BASH="${DLRM_PATH}/dlrm_venv/bin/activate"
TORCHREC_SCRIPT="${DLRM_PATH}/tests/dlrm_main_torchrec_single.py"
CUSTOM_SCRIPT="${DLRM_PATH}/tests/dlrm_main_single_day.py"

use_torchrec=false
dataset_size=$DEFAULT_DATASET_SIZE
processed_dataset_path=$DEFAULT_PROCESSED_DATASET_PATH
batch_size=$DEFAULT_BATCH_SIZE
learning_rate=$DEFAULT_LEARNING_RATE
epochs=$DEFAULT_EPOCHS

show_help() {
    echo "DLRM Training Script with Performance Metrics"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help                  Show this help message and exit"
    echo ""
    echo "Configuration (defaults used if not specified):"
    echo "  --dataset-size SIZE         Dataset size (default: $DEFAULT_DATASET_SIZE)"
    echo "  --dataset-path PATH         Processed dataset path (default: $DEFAULT_PROCESSED_DATASET_PATH)"
    echo "  --batch-size SIZE           Batch size (default: $DEFAULT_BATCH_SIZE)"
    echo "  --learning-rate RATE        Learning rate (default: $DEFAULT_LEARNING_RATE)"
    echo "  --epochs COUNT              Number of epochs (default: $DEFAULT_EPOCHS)"
    echo ""
    echo "Mode Selection (choose one):"
    echo "  --torchrec                  Use TorchRec baseline (default: custom recstore)"
    echo "  --custom                    Use custom recstore (default behavior)"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --torchrec)
            use_torchrec=true
            shift
            ;;
        --custom)
            use_torchrec=false
            shift
            ;;
        --dataset-size)
            if [[ -n "$2" && "$2" != -* ]]; then
                dataset_size="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        --dataset-path)
            if [[ -n "$2" && "$2" != -* ]]; then
                processed_dataset_path="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        --batch-size)
            if [[ -n "$2" && "$2" != -* ]]; then
                batch_size="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        --learning-rate)
            if [[ -n "$2" && "$2" != -* ]]; then
                learning_rate="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        --epochs)
            if [[ -n "$2" && "$2" != -* ]]; then
                epochs="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1" >&2
            show_help
            exit 1
            ;;
    esac
done

if [ "$use_torchrec" = true ]; then
    script_to_run="$TORCHREC_SCRIPT"
    mode="TorchRec"
else
    script_to_run="$CUSTOM_SCRIPT"
    mode="RecStore"
fi

echo "=========================================="
echo "DLRM Training Configuration"
echo "=========================================="
echo "Mode:                     $mode"
echo "Dataset Size:             $dataset_size"
echo "Dataset Path:             $processed_dataset_path"
echo "Batch Size:               $batch_size"
echo "Learning Rate:            $learning_rate"
echo "Epochs:                   $epochs"
echo "Script Path:              $script_to_run"
echo "Env. Path:                $VENV_BASH"
echo "=========================================="

source ${VENV_BASH}

if [[ ! -f "$script_to_run" ]]; then
    echo "Error: Script not found at $script_to_run" >&2
    exit 1
fi

if [ ! -d "$processed_dataset_path" ]; then
    echo "Error: Processed dataset not found at $processed_dataset_path"
    echo "Please run the preprocessing script first:"
    echo "bash scripts/process_single_day.sh <raw_data_dir> $processed_dataset_path"
    exit 1
fi

required_files=("day_0_dense.npy" "day_0_sparse.npy" "day_0_labels.npy")
for file in "${required_files[@]}"; do
    if [ ! -f "$processed_dataset_path/$file" ]; then
        echo "Error: Required file $file not found in $processed_dataset_path"
        exit 1
    fi
done

echo "✓ All required data files found"
echo ""

start_time=$(date +%s.%N)
start_seconds=$(date +%s)

echo "Starting training..."
torchrun --nnodes 1 \
    --nproc_per_node 1 \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost \
    --rdzv_id 54321 \
    --role trainer $script_to_run \
    --single_day_mode \
    --in_memory_binary_criteo_path $processed_dataset_path \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --epochs $epochs \
    --pin_memory \
    --mmap_mode \
    --embedding_dim 128 \
    --adagrad > training_output.$dataset_size.$mode.$(date +%Y%m%d%H%M%S).log 2>&1


end_time=$(date +%s.%N)
end_seconds=$(date +%s)

duration=$((end_seconds - start_seconds))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo "=========================================="
echo "Training Execution Summary"
echo "=========================================="
echo "Start Time:               $(date -d "@$(echo $start_time | cut -d. -f1)" '+%Y-%m-%d %H:%M:%S')"
echo "End Time:                 $(date -d "@$(echo $end_time | cut -d. -f1)" '+%Y-%m-%d %H:%M:%S')"
echo "Total Duration:           $(printf "%02d" $hours):$(printf "%02d" $minutes):$(printf "%02d" $seconds)"
echo "=========================================="
