#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025 RecStore Choimoe. All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Enhanced by Choimoe:
# - Supports loading an arbitrary? number of days from the dataset

display_help() {
   echo "Two command line arguments are required."
   echo "Example usage:"
   echo "bash process_single_day.sh \\"
   echo "./criteo_1tb/raw_input_dataset_dir \\"
   echo "./criteo_1tb/processed_output_dataset_dir"
   exit 1
}

[ -z "$1" ] && display_help
[ -z "$2" ] && display_help

raw_tsv_criteo_files_dir=$(readlink -m "$1")

output_dir=$(readlink -m "$2")

echo "Processing single day data..."
echo "Input directory: $raw_tsv_criteo_files_dir"
echo "Output directory: $output_dir"

day_0_file="$raw_tsv_criteo_files_dir/day_0.tsv"
if [ ! -f "$day_0_file" ]; then
    echo "Error: day_0 file not found in $raw_tsv_criteo_files_dir"
    echo "Please ensure you have downloaded and uncompressed day_0.gz"
    exit 1
fi

mkdir -p "$output_dir"

echo "Step 1: Converting TSV to numpy format..."
date

python -m torchrec.datasets.scripts.npy_preproc_criteo \
    --input_dir "$raw_tsv_criteo_files_dir" \
    --output_dir "$output_dir" \
    --dataset_name criteo_1tb

if [ $? -ne 0 ]; then
    echo "Error: Failed to convert TSV to numpy format"
    exit 1
fi

echo "Step 2: Converting sparse indices to contiguous indices..."
date

python -m torchrec.datasets.scripts.contiguous_preproc_criteo \
    --input_dir "$output_dir" \
    --output_dir "$output_dir" \
    --frequency_threshold 0

if [ $? -ne 0 ]; then
    echo "Error: Failed to convert sparse indices"
    exit 1
fi

for i in 0
do
   name="$output_dir/day_$i""_sparse_contig_freq.npy"
   renamed="$output_dir/day_$i""_sparse.npy"
   echo "Renaming $name to $renamed"
   mv "$name" "$renamed"
done

echo "Step 3: Shuffling the dataset..."
date

python -m torchrec.datasets.scripts.shuffle_preproc_criteo \
    --input_dir_labels_and_dense "$output_dir" \
    --input_dir_sparse "$output_dir" \
    --output_dir_shuffled "$output_dir" \
    --random_seed 0 \
    --days 1

if [ $? -ne 0 ]; then
    echo "Error: Failed to shuffle dataset"
    exit 1
fi

echo "Processing completed successfully!"
echo "Output files in $output_dir:"
ls -la "$output_dir"

echo "Verifying output files..."
required_files=("day_0_dense.npy" "day_0_sparse.npy" "day_0_labels.npy")
for file in "${required_files[@]}"; do
    if [ -f "$output_dir/$file" ]; then
        echo "✓ $file exists"
        file_size=$(du -h "$output_dir/$file" | cut -f1)
        echo "  Size: $file_size"
    else
        echo "✗ $file missing"
        exit 1
    fi
done

echo ""
echo "Single day data processing completed successfully!"
echo "You can now use the processed data for training with:"
echo "python dlrm_main_single_day.py --single_day_mode --in_memory_binary_criteo_path $output_dir" 