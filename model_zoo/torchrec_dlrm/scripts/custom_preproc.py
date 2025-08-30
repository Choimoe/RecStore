#!/usr/bin/env python3
import argparse
import numpy as np
import os
from tqdm import tqdm

def process_criteo_data(input_file, output_dir):
    print(f"Processing {input_file}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    dense_features = []
    sparse_features = []
    labels = []
    
    total_lines = 0
    error_lines = 0
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(tqdm(f, desc="Processing lines")):
            try:
                parts = line.strip().split('\t')
                
                if len(parts) != 40:
                    print(f"Warning: Line {line_num + 1} has {len(parts)} columns, expected 40")
                    error_lines += 1
                    continue
                
                label = int(parts[0])
                labels.append(label)
                
                dense = []
                for i in range(1, 14):
                    try:
                        val = int(parts[i]) if parts[i] else 0
                        dense.append(val)
                    except ValueError:
                        dense.append(0)
                dense_features.append(dense)
                
                sparse = []
                for i in range(14, 40):
                    try:
                        if parts[i]:
                            val = int(parts[i], 16)
                        else:
                            val = 0
                        sparse.append(val)
                    except ValueError:
                        sparse.append(0)
                sparse_features.append(sparse)
                
                total_lines += 1
                
            except Exception as e:
                print(f"Error processing line {line_num + 1}: {e}")
                error_lines += 1
                continue
    
    print(f"Processed {total_lines} lines successfully")
    if error_lines > 0:
        print(f"Encountered {error_lines} errors")
    
    dense_array = np.array(dense_features, dtype=np.int32)
    sparse_array = np.array(sparse_features, dtype=np.int32)
    labels_array = np.array(labels, dtype=np.int32)
    
    print(f"Dense features shape: {dense_array.shape}")
    print(f"Sparse features shape: {sparse_array.shape}")
    print(f"Labels shape: {labels_array.shape}")
    
    dense_file = os.path.join(output_dir, "day_0_dense.npy")
    sparse_file = os.path.join(output_dir, "day_0_sparse.npy")
    labels_file = os.path.join(output_dir, "day_0_labels.npy")
    
    print(f"Saving dense features to {dense_file}")
    np.save(dense_file, dense_array)
    
    print(f"Saving sparse features to {sparse_file}")
    np.save(sparse_file, sparse_array)
    
    print(f"Saving labels to {labels_file}")
    np.save(labels_file, labels_array)
    
    for file_path in [dense_file, sparse_file, labels_file]:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"{os.path.basename(file_path)}: {size_mb:.2f} MB")
    
    print("Preprocessing completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Custom Criteo data preprocessing")
    parser.add_argument("--input_file", required=True, help="Input TSV file path")
    parser.add_argument("--output_dir", required=True, help="Output directory for numpy files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found")
        return
    
    process_criteo_data(args.input_file, args.output_dir)

if __name__ == "__main__":
    main() 