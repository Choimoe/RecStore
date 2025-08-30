#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025 RecStore Choimoe. All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Enhanced by Choimoe:
# - Supports loading an arbitrary number of days from the dataset
# - Falls back to using the last available day as validation set when insufficient data

import argparse
import os
from typing import List

import torch
from torch import distributed as dist
from torch.utils.data import DataLoader

from torchrec.datasets.criteo import (
    CAT_FEATURE_COUNT,
    InMemoryBinaryCriteoIterDataPipe,
    MultiHotCriteoIterDataPipe,
)

STAGES = ["train", "val", "test"]


def _get_random_dataloader(
    args: argparse.Namespace,
    stage: str,
) -> DataLoader:
    from torchrec.datasets.random import RandomRecDataset

    if stage == "train":
        dataset = RandomRecDataset(
            keys=args.num_embeddings_per_feature,
            batch_size=args.batch_size,
            hash_size=args.num_embeddings_per_feature,
            hash_dtype=torch.int32,
            ids_per_feature=1,
            num_dense=13,
        )
    else:
        dataset = RandomRecDataset(
            keys=args.num_embeddings_per_feature,
            batch_size=args.test_batch_size or args.batch_size,
            hash_size=args.num_embeddings_per_feature,
            hash_dtype=torch.int32,
            ids_per_feature=1,
            num_dense=13,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        pin_memory=args.pin_memory,
        collate_fn=lambda x: x,
    )
    return dataloader


def _get_single_day_dataloader(
    args: argparse.Namespace,
    stage: str,
) -> DataLoader:
    if args.in_memory_binary_criteo_path is None:
        raise ValueError("--in_memory_binary_criteo_path must be specified for single day training")
    
    dir_path = args.in_memory_binary_criteo_path
    sparse_part = "sparse.npy"
    datapipe = InMemoryBinaryCriteoIterDataPipe
    
    day_0_dense = os.path.join(dir_path, "day_0_dense.npy")
    day_0_sparse = os.path.join(dir_path, "day_0_sparse.npy")
    day_0_labels = os.path.join(dir_path, "day_0_labels.npy")
    
    if not all(os.path.exists(f) for f in [day_0_dense, day_0_sparse, day_0_labels]):
        raise FileNotFoundError(f"Day 0 files not found in {dir_path}. Please ensure you have processed day_0 data.")
    
    if stage == "train":
        stage_files: List[List[str]] = [
            [day_0_dense],
            [day_0_sparse],
            [day_0_labels],
        ]
        train_ratio = 0.8
    elif stage in ["val", "test"]:
        stage_files: List[List[str]] = [
            [day_0_dense],
            [day_0_sparse],
            [day_0_labels],
        ]
        train_ratio = 0.8
    else:
        raise ValueError(f"Invalid stage: {stage}")
    
    if stage in ["val", "test"] and args.test_batch_size is not None:
        batch_size = args.test_batch_size
    else:
        batch_size = args.batch_size
    
    dataloader = DataLoader(
        datapipe(
            stage,
            *stage_files,
            batch_size=batch_size,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            drop_last=args.drop_last_training_batch if stage == "train" else False,
            shuffle_batches=args.shuffle_batches,
            shuffle_training_set=args.shuffle_training_set,
            shuffle_training_set_random_seed=args.seed,
            mmap_mode=args.mmap_mode,
            hashes=(
                args.num_embeddings_per_feature
                if args.num_embeddings is None
                else ([args.num_embeddings] * CAT_FEATURE_COUNT)
            ),
            single_day_mode=True,
            train_ratio=train_ratio,
        ),
        batch_size=None,
        pin_memory=args.pin_memory,
        collate_fn=lambda x: x,
    )
    return dataloader


def get_dataloader(args: argparse.Namespace, backend: str, stage: str) -> DataLoader:
    stage = stage.lower()
    if stage not in STAGES:
        raise ValueError(f"Supplied stage was {stage}. Must be one of {STAGES}.")

    args.pin_memory = (
        (backend == "nccl") if not hasattr(args, "pin_memory") else args.pin_memory
    )

    if (
        args.in_memory_binary_criteo_path is None
        and args.synthetic_multi_hot_criteo_path is None
    ):
        return _get_random_dataloader(args, stage)
    else:
        if hasattr(args, 'single_day_mode') and args.single_day_mode:
            return _get_single_day_dataloader(args, stage)
        else:
            return _get_in_memory_dataloader(args, stage)


def _get_in_memory_dataloader(
    args: argparse.Namespace,
    stage: str,
) -> DataLoader:
    if args.in_memory_binary_criteo_path is not None:
        dir_path = args.in_memory_binary_criteo_path
        sparse_part = "sparse.npy"
        datapipe = InMemoryBinaryCriteoIterDataPipe
    else:
        dir_path = args.synthetic_multi_hot_criteo_path
        sparse_part = "sparse_multi_hot.npz"
        datapipe = MultiHotCriteoIterDataPipe

    if args.dataset_name == "criteo_kaggle":
        # criteo_kaggle has no validation set, so use 2nd half of training set for now.
        # Setting stage to "test" will get the 2nd half of the dataset.
        # Setting root_name to "train" reads from the training set file.
        (root_name, stage) = (
            ("train", "train") if stage == "train" else ("train", "test")
        )
        stage_files: List[List[str]] = [
            [os.path.join(dir_path, f"{root_name}_dense.npy")],
            [os.path.join(dir_path, f"{root_name}_{sparse_part}")],
            [os.path.join(dir_path, f"{root_name}_labels.npy")],
        ]
    # criteo_1tb code path uses below two conditionals
    elif stage == "train":
        available_days = []
        for i in range(24):
            day_files = [
                os.path.join(dir_path, f"day_{i}_dense.npy"),
                os.path.join(dir_path, f"day_{i}_{sparse_part}"),
                os.path.join(dir_path, f"day_{i}_labels.npy")
            ]
            if all(os.path.exists(f) for f in day_files):
                available_days.append(i)
        
        if not available_days:
            raise FileNotFoundError(f"No complete day files found in {dir_path}")
        
        train_days = available_days[:-1] if len(available_days) > 1 else available_days
        
        stage_files: List[List[str]] = [
            [os.path.join(dir_path, f"day_{i}_dense.npy") for i in train_days],
            [os.path.join(dir_path, f"day_{i}_{sparse_part}") for i in train_days],
            [os.path.join(dir_path, f"day_{i}_labels.npy") for i in train_days],
        ]
    elif stage in ["val", "test"]:
        available_days = []
        for i in range(24):
            day_files = [
                os.path.join(dir_path, f"day_{i}_dense.npy"),
                os.path.join(dir_path, f"day_{i}_{sparse_part}"),
                os.path.join(dir_path, f"day_{i}_labels.npy")
            ]
            if all(os.path.exists(f) for f in day_files):
                available_days.append(i)
        
        if not available_days:
            raise FileNotFoundError(f"No complete day files found in {dir_path}")
        
        val_day = available_days[-1]
        
        stage_files: List[List[str]] = [
            [os.path.join(dir_path, f"day_{val_day}_dense.npy")],
            [os.path.join(dir_path, f"day_{val_day}_{sparse_part}")],
            [os.path.join(dir_path, f"day_{val_day}_labels.npy")],
        ]
    
    if stage in ["val", "test"] and args.test_batch_size is not None:
        batch_size = args.test_batch_size
    else:
        batch_size = args.batch_size
    
    dataloader = DataLoader(
        datapipe(
            stage,
            *stage_files,
            batch_size=batch_size,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            drop_last=args.drop_last_training_batch if stage == "train" else False,
            shuffle_batches=args.shuffle_batches,
            shuffle_training_set=args.shuffle_training_set,
            shuffle_training_set_random_seed=args.seed,
            mmap_mode=args.mmap_mode,
            hashes=(
                args.num_embeddings_per_feature
                if args.num_embeddings is None
                else ([args.num_embeddings] * CAT_FEATURE_COUNT)
            ),
        ),
        batch_size=None,
        pin_memory=args.pin_memory,
        collate_fn=lambda x: x,
    )
    return dataloader 