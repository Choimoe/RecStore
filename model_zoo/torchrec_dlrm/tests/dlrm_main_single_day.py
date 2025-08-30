#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, List, Optional

import torch
import torchmetrics as metrics
from pyre_extensions import none_throws
from torch import distributed as dist
from torch.utils.data import DataLoader
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
# from torchrec.models.dlrm import DLRM, DLRM_DCN, DLRM_Projection, DLRMTrain
# [yinj]
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from tqdm import tqdm

RECSTORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src'))
DLRM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if RECSTORE_PATH not in sys.path:
    sys.path.insert(0, RECSTORE_PATH)
    sys.path.insert(0, DLRM_PATH)

from dlrm import DLRM, DLRM_DCN, DLRM_Projection, DLRMTrain
from python.pytorch.torchrec.EmbeddingBag import RecStoreEmbeddingBagCollection

try:
    from data.custom_dataloader import get_dataloader
    from lr_scheduler import LRPolicyScheduler
    from multi_hot import Multihot, RestartableMap
except ImportError:
    try:
        from ..data.custom_dataloader import get_dataloader
        from ..lr_scheduler import LRPolicyScheduler
        from ..multi_hot import Multihot, RestartableMap
    except ImportError:
        print("Warning: Could not import custom dataloader modules")
        def get_dataloader(args, backend, stage):
            raise NotImplementedError("Please ensure custom dataloader modules are available")

TRAIN_PIPELINE_STAGES = 3  # Number of stages in TrainPipelineSparseDist.


class InteractionType(Enum):
    ORIGINAL = "original"
    DCN = "dcn"
    PROJECTION = "projection"

    def __str__(self):
        return self.value


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer for single day data")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--drop_last_training_batch",
        dest="drop_last_training_batch",
        action="store_true",
        help="Drop last non-full training batch",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="batch size to use for validation and testing",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=None,
        help="number of embeddings (hashes) per sparse feature",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="comma separated list of number of embeddings per sparse feature",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="number of embedding dimensions",
    )
    parser.add_argument(
        "--interaction_type",
        type=InteractionType,
        choices=list(InteractionType),
        default=InteractionType.ORIGINAL,
        help="type of interaction layer to use",
    )
    parser.add_argument(
        "--dcn_num_layers",
        type=int,
        default=3,
        help="number of DCN layers",
    )
    parser.add_argument(
        "--dcn_low_rank_dim",
        type=int,
        default=512,
        help="low rank dimension for DCN",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="1024,1024,512,256,1",
        help="comma separated layer sizes for over arch",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,128",
        help="comma separated layer sizes for dense arch",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for reproducibility",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory for faster GPU transfer",
    )
    parser.add_argument(
        "--mmap_mode",
        dest="mmap_mode",
        action="store_true",
        help="Use memory mapping for loading data",
    )
    parser.add_argument(
        "--in_memory_binary_criteo_path",
        type=str,
        default=None,
        help="Path to preprocessed Criteo dataset",
    )
    parser.add_argument(
        "--synthetic_multi_hot_criteo_path",
        type=str,
        default=None,
        help="Path to synthetic multi-hot Criteo dataset",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="criteo_1tb",
        help="Dataset name (criteo_1tb or criteo_kaggle)",
    )
    parser.add_argument(
        "--shuffle_batches",
        dest="shuffle_batches",
        action="store_true",
        help="Shuffle batches",
    )
    parser.add_argument(
        "--shuffle_training_set",
        dest="shuffle_training_set",
        action="store_true",
        help="Shuffle training set",
    )
    parser.add_argument(
        "--validation_freq_within_epoch",
        type=int,
        default=None,
        help="Validation frequency within epoch",
    )
    parser.add_argument(
        "--adagrad",
        dest="adagrad",
        action="store_true",
        help="Use Adagrad optimizer",
    )
    parser.add_argument(
        "--single_day_mode",
        dest="single_day_mode",
        action="store_true",
        help="Enable single day data training mode",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training (rest for validation) in single day mode",
    )
    
    args = parser.parse_args(argv)
    
    if args.single_day_mode:
        if args.in_memory_binary_criteo_path is None:
            raise ValueError("--in_memory_binary_criteo_path must be specified for single day mode")
        
        if args.num_embeddings_per_feature is None:
            args.num_embeddings_per_feature = "40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36"
        
        if not args.adagrad:
            args.learning_rate = 0.005
            args.adagrad = True
        
        print(f"Single day mode enabled. Training with day_0 data only.")
        print(f"Training ratio: {args.train_ratio}")
        print(f"Validation ratio: {1 - args.train_ratio}")
    
    return args


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", dist.get_rank())
    else:
        dist.init_process_group(backend="gloo")
        device = torch.device("cpu")
    
    print(f"Distributed training initialized:")
    print(f"  Rank: {dist.get_rank()}")
    print(f"  World size: {dist.get_world_size()}")
    print(f"  Device: {device}")
    
    train_dataloader = get_dataloader(args, "nccl" if torch.cuda.is_available() else "gloo", "train")
    val_dataloader = get_dataloader(args, "nccl" if torch.cuda.is_available() else "gloo", "val")
    
    def custom_collate(batch):
        if not batch:
            return batch
        
        dense_features = []
        sparse_features = []
        labels = []
        
        for dense, sparse, label in batch:
            dense_features.append(torch.as_tensor(dense, dtype=torch.float32))
            sparse_features.append(sparse)
            labels.append(torch.as_tensor(label, dtype=torch.float32))
        
        dense_batch = torch.stack(dense_features)
        
        from torchrec import KeyedJaggedTensor
        feature_names = DEFAULT_CAT_NAMES
        
        sparse_mat = torch.stack([s.to(torch.long) for s in sparse_features], dim=0)
        B = sparse_mat.shape[0]
        values_list = [sparse_mat[:, i] for i in range(26)]
        values = torch.cat(values_list, dim=0)
        one_lengths = torch.ones(B, dtype=torch.int32, device=values.device)
        lengths_list = [one_lengths for _ in range(26)]
        lengths = torch.cat(lengths_list, dim=0)
        sparse_batch = KeyedJaggedTensor.from_lengths_sync(
            keys=feature_names,
            values=values,
            lengths=lengths,
        )
        
        labels_batch = torch.stack(labels)
        
        return dense_batch, sparse_batch, labels_batch
    
    train_dataloader = DataLoader(
        train_dataloader.dataset,
        batch_size=train_dataloader.batch_size,
        shuffle=True,
        drop_last=args.drop_last_training_batch,
        pin_memory=args.pin_memory,
        collate_fn=custom_collate,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataloader.dataset,
        batch_size=val_dataloader.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=args.pin_memory,
        collate_fn=custom_collate,
        num_workers=0
    )
    
    if args.num_embeddings_per_feature is not None:
        num_embeddings_per_feature = [
            int(x) for x in args.num_embeddings_per_feature.split(",")
        ]
    else:
        num_embeddings_per_feature = [args.num_embeddings] * 26
    
    eb_configs = [
        {
            "name": f"t_{feature_name}",
            "num_embeddings": num_embeddings_per_feature[feature_idx],
            "embedding_dim": args.embedding_dim,
            "feature_names": [feature_name],
        }
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]
    embedding_bag_collection = RecStoreEmbeddingBagCollection(eb_configs)
    
    if args.interaction_type == InteractionType.DCN:
        model = DLRM_DCN(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=13,
            dense_arch_layer_sizes=[int(x) for x in args.dense_arch_layer_sizes.split(",")],
            over_arch_layer_sizes=[int(x) for x in args.over_arch_layer_sizes.split(",")],
            dcn_num_layers=args.dcn_num_layers,
            dcn_low_rank_dim=args.dcn_low_rank_dim,
        )
    else:
        model = DLRM(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=13,
            dense_arch_layer_sizes=[int(x) for x in args.dense_arch_layer_sizes.split(",")],
            over_arch_layer_sizes=[int(x) for x in args.over_arch_layer_sizes.split(",")],
        )
    
    model = model.to(device)
    
    if args.adagrad:
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    auroc = metrics.AUROC(task="binary")
    
    model.train()
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        
        train_loss = 0.0
        train_auroc = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            dense_features = batch[0].to(device)
            sparse_features = batch[1].to(device)
            labels = batch[2].to(device)
            
            optimizer.zero_grad()
            outputs = model(dense_features, sparse_features)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            auroc_score = auroc(outputs.squeeze(), labels)
            train_auroc += auroc_score.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}, AUROC = {auroc_score.item():.4f}")
        
        avg_train_loss = train_loss / num_batches
        avg_train_auroc = train_auroc / num_batches
        
        print(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}, Training AUROC: {avg_train_auroc:.4f}")
        
        model.eval()
        val_loss = 0.0
        val_auroc = 0.0
        val_num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                dense_features = batch[0].to(device)
                sparse_features = batch[1].to(device)
                labels = batch[2].to(device)
                
                outputs = model(dense_features, sparse_features)
                loss = criterion(outputs, labels.float().unsqueeze(1))
                
                val_loss += loss.item()
                auroc_score = auroc(outputs.squeeze(), labels)
                val_auroc += auroc_score.item()
                val_num_batches += 1
        
        avg_val_loss = val_loss / val_num_batches
        avg_val_auroc = val_auroc / val_num_batches
        
        print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}, Validation AUROC: {avg_val_auroc:.4f}")
        
        model.train()
        auroc.reset()
    
    print("Training completed!")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
