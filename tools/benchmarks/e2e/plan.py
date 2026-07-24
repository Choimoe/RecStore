from __future__ import annotations

from pathlib import Path

from .common import E2ELane, ExperimentPlan, PlanOverrides


def build_plan(profile: str, output_root: Path, overrides: PlanOverrides | None = None) -> ExperimentPlan:
    overrides = overrides or PlanOverrides()
    normalized = profile.strip().lower()
    if normalized == "smoke":
        data_rows = (4096,)
        batch_sizes = (256,)
        num_embeddings = (10000,)
        embedding_dims = (128,)
        steps = 3
        warmup_steps = 1
        repeat = 1
    elif normalized == "pilot":
        data_rows = (4096, 32768)
        batch_sizes = (512, 1024)
        num_embeddings = (50000, 200000)
        embedding_dims = (64, 128)
        steps = 20
        warmup_steps = 3
        repeat = 1
    elif normalized == "stress":
        data_rows = (131072, 524288)
        batch_sizes = (1024, 4096)
        num_embeddings = (200000, 800000)
        embedding_dims = (64, 128)
        steps = 30
        warmup_steps = 5
        repeat = 1
    elif normalized == "full":
        data_rows = (4096, 32768, 131072, 524288)
        batch_sizes = (512, 1024, 2048, 4096)
        num_embeddings = (50000, 200000, 800000)
        embedding_dims = (64, 128)
        steps = 60
        warmup_steps = 5
        repeat = 3
    else:
        raise ValueError(f"unknown profile: {profile}")

    lanes = [
        E2ELane(
            slug="torchrec-hbm-1p",
            label="TorchRec-HBM-1proc",
            backend="torchrec",
            nproc_per_node=1,
            torchrec_memory_mode="hbm",
        ),
        E2ELane(
            slug="torchrec-uvm-1p",
            label="TorchRec-UVMCache-1proc",
            backend="torchrec",
            nproc_per_node=1,
            torchrec_memory_mode="uvm_caching",
        ),
        E2ELane(
            slug="recstore-brpc-pet-1p",
            label="RecStore-BRPC-PET-1proc",
            backend="recstore",
            ps_type="BRPC",
            recstore_index_type="DRAM_PET_HASH",
            ps_kv_backend="recstore_dram",
            nproc_per_node=1,
        ),
        E2ELane(
            slug="recstore-rdma-pet-1p",
            label="RecStore-RDMA-PET-1proc",
            backend="recstore",
            ps_type="RDMA",
            recstore_index_type="DRAM_PET_HASH",
            ps_kv_backend="recstore_dram",
            nproc_per_node=1,
        ),
        E2ELane(
            slug="recstore-brpc-eh-1p",
            label="RecStore-BRPC-EH-1proc",
            backend="recstore",
            ps_type="BRPC",
            recstore_index_type="DRAM_EXTENDIBLE_HASH",
            ps_kv_backend="recstore_dram",
            nproc_per_node=1,
        ),
        E2ELane(
            slug="recstore-grpc-pet-1p",
            label="RecStore-GRPC-PET-1proc",
            backend="recstore",
            ps_type="GRPC",
            recstore_index_type="DRAM_PET_HASH",
            ps_kv_backend="recstore_dram",
            nproc_per_node=1,
        ),
        E2ELane(
            slug="recstore-brpc-pet-prefetch4-1p",
            label="RecStore-BRPC-PET-prefetch4-1proc",
            backend="recstore",
            ps_type="BRPC",
            recstore_index_type="DRAM_PET_HASH",
            ps_kv_backend="recstore_dram",
            nproc_per_node=1,
            prefetch_depth=4,
        ),
        E2ELane(
            slug="torchrec-hbm-2p",
            label="TorchRec-HBM-2proc",
            backend="torchrec",
            nproc_per_node=2,
            torchrec_memory_mode="hbm",
        ),
        E2ELane(
            slug="recstore-local-shm-pet-2p",
            label="RecStore-LOCAL_SHM-PET-2proc",
            backend="recstore",
            ps_type="LOCAL_SHM",
            recstore_index_type="DRAM_PET_HASH",
            ps_kv_backend="recstore_dram",
            nproc_per_node=2,
            single_node_ps_backend="local_shm",
        ),
        E2ELane(
            slug="recstore-brpc-pet-2p",
            label="RecStore-BRPC-PET-2proc",
            backend="recstore",
            ps_type="BRPC",
            recstore_index_type="DRAM_PET_HASH",
            ps_kv_backend="recstore_dram",
            nproc_per_node=2,
        ),
    ]
    if overrides.include_ablation_lanes:
        lanes.extend(
            [
                E2ELane(
                    slug="recstore-brpc-map-1p",
                    label="RecStore-BRPC-MAP-1proc",
                    backend="recstore",
                    ps_type="BRPC",
                    recstore_index_type="DRAM_UNORDERED_MAP",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                ),
                E2ELane(
                    slug="recstore-grpc-eh-1p",
                    label="RecStore-GRPC-EH-1proc",
                    backend="recstore",
                    ps_type="GRPC",
                    recstore_index_type="DRAM_EXTENDIBLE_HASH",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                ),
                E2ELane(
                    slug="recstore-local-shm-pet-1p",
                    label="RecStore-LOCAL_SHM-PET-1proc",
                    backend="recstore",
                    ps_type="LOCAL_SHM",
                    recstore_index_type="DRAM_PET_HASH",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                ),
                E2ELane(
                    slug="recstore-rdma-eh-1p",
                    label="RecStore-RDMA-EH-1proc",
                    backend="recstore",
                    ps_type="RDMA",
                    recstore_index_type="DRAM_EXTENDIBLE_HASH",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                ),
                E2ELane(
                    slug="recstore-rdma-map-1p",
                    label="RecStore-RDMA-MAP-1proc",
                    backend="recstore",
                    ps_type="RDMA",
                    recstore_index_type="DRAM_UNORDERED_MAP",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                ),
                E2ELane(
                    slug="recstore-brpc-pet-prefetch1-1p",
                    label="RecStore-BRPC-PET-prefetch1-1proc",
                    backend="recstore",
                    ps_type="BRPC",
                    recstore_index_type="DRAM_PET_HASH",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                    prefetch_depth=1,
                ),
                E2ELane(
                    slug="recstore-rdma-pet-prefetch1-1p",
                    label="RecStore-RDMA-PET-prefetch1-1proc",
                    backend="recstore",
                    ps_type="RDMA",
                    recstore_index_type="DRAM_PET_HASH",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                    prefetch_depth=1,
                ),
                E2ELane(
                    slug="recstore-brpc-pet-prefetch8-1p",
                    label="RecStore-BRPC-PET-prefetch8-1proc",
                    backend="recstore",
                    ps_type="BRPC",
                    recstore_index_type="DRAM_PET_HASH",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                    prefetch_depth=8,
                ),
                E2ELane(
                    slug="recstore-rdma-pet-prefetch4-1p",
                    label="RecStore-RDMA-PET-prefetch4-1proc",
                    backend="recstore",
                    ps_type="RDMA",
                    recstore_index_type="DRAM_PET_HASH",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                    prefetch_depth=4,
                ),
            ]
        )
    if overrides.only_lanes:
        allowed = set(overrides.only_lanes)
        known = {lane.slug for lane in lanes}
        unknown = sorted(allowed - known)
        if unknown:
            raise ValueError(f"unknown lane slug(s): {', '.join(unknown)}")
        lanes = [lane for lane in lanes if lane.slug in allowed]
    return ExperimentPlan(
        profile=normalized,
        output_root=output_root,
        data_rows=overrides.data_rows or data_rows,
        batch_sizes=overrides.batch_sizes or batch_sizes,
        num_embeddings=overrides.num_embeddings or num_embeddings,
        embedding_dims=overrides.embedding_dims or embedding_dims,
        steps=overrides.steps if overrides.steps is not None else steps,
        warmup_steps=overrides.warmup_steps if overrides.warmup_steps is not None else warmup_steps,
        repeat=overrides.repeat if overrides.repeat is not None else repeat,
        lanes=tuple(lanes),
    )
