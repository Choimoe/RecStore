#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025 RecStore Choimoe. All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Enhanced by Choimoe:
# - Supports loading an arbitrary? number of days from the dataset

# Usage:
# nohup bash scripts/process_single_day.sh ./partial_data ./processed_day_0_data > preprocess_data.log 2>&1 &

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

day_0_file="$raw_tsv_criteo_files_dir/day_0"
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

python -c "
import numpy as np
import os

input_dir = '$output_dir'
output_dir = '$output_dir'

day_0_sparse_file = os.path.join(input_dir, 'day_0_sparse.npy')

if not os.path.exists(day_0_sparse_file):
    print(f'Error: {day_0_sparse_file} not found')
    exit(1)

print(f'Processing {day_0_sparse_file}...')

sparse_data = np.load(day_0_sparse_file)

unique_features = np.unique(sparse_data)

feature_to_contig = {feature: idx for idx, feature in enumerate(unique_features)}

contig_sparse_data = np.array([[feature_to_contig[feature] for feature in row] for row in sparse_data])

output_file = os.path.join(output_dir, 'day_0_sparse_contig_freq.npy')
np.save(output_file, contig_sparse_data)

print(f'Converted sparse indices saved to {output_file}')
print(f'Original unique features: {len(unique_features)}')
print(f'Contiguous indices range: 0 to {len(unique_features)-1}')
"

if [ $? -ne 0 ]; then
    echo "Error: Failed to convert sparse indices"
    exit 1
fi

echo "Renaming files..."
mv "$output_dir/day_0_sparse_contig_freq.npy" "$output_dir/day_0_sparse.npy"

echo "Step 3: Shuffling the dataset (single day only)..."

date

python -c "
import numpy as np
import os
import random

input_dir = '$output_dir'
output_dir = '$output_dir'

random.seed(0)
np.random.seed(0)

print('Loading day_0 data...')

dense_file = os.path.join(input_dir, 'day_0_dense.npy')
sparse_file = os.path.join(input_dir, 'day_0_sparse.npy')
labels_file = os.path.join(input_dir, 'day_0_labels.npy')

if not all(os.path.exists(f) for f in [dense_file, sparse_file, labels_file]):
    print('Error: Required day_0 files not found')
    exit(1)

dense_data = np.load(dense_file)
sparse_data = np.load(sparse_file)
labels_data = np.load(labels_file)

print(f'Loaded data shapes:')
print(f'  Dense: {dense_data.shape}')
print(f'  Sparse: {sparse_data.shape}')
print(f'  Labels: {labels_data.shape}')

num_samples = len(dense_data)
indices = list(range(num_samples))
random.shuffle(indices)

print(f'Shuffling {num_samples} samples...')

dense_shuffled = dense_data[indices]
sparse_shuffled = sparse_data[indices]
labels_shuffled = labels_data[indices]

np.save(os.path.join(output_dir, 'day_0_dense.npy'), dense_shuffled)
np.save(os.path.join(output_dir, 'day_0_sparse.npy'), sparse_shuffled)
np.save(os.path.join(output_dir, 'day_0_labels.npy'), labels_shuffled)

print('Shuffling completed successfully!')
"

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
        
        python -c "
import numpy as np
data = np.load('$output_dir/$file')
print(f'    Shape: {data.shape}')
print(f'    Data type: {data.dtype}')
"
    else
        echo "✗ $file missing"
        exit 1
    fi
done

echo ""
echo "Single day data processing completed successfully!"
