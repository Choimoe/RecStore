"""Explicit two-rank CUDA metadata oracle; this script does not launch workers.

Launch one process per host with torchrun, selecting the hosts, rendezvous,
Python runtime, CUDA_VISIBLE_DEVICES, and network interfaces externally.
Provide --output-dir on each host. No PS server or embedding data is required.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time
from unittest import mock

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Explicit directory for this rank's correctness report")
    parser.add_argument("--device", type=int, default=None,
                        help="Visible CUDA device index; defaults to LOCAL_RANK")
    parser.add_argument("--init-method", default="env://",
                        help="Distributed rendezvous, normally supplied by torchrun environment")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--random-cases", type=int, default=18)
    parser.add_argument("--seed", type=int, default=410)
    parser.add_argument("--configs", default="1:9,3:17,17:5,26:1024",
                        help="Comma-separated feature_count:max_batch_size test cases")
    args = parser.parse_args()
    if args.random_cases < 0 or args.timeout_seconds < 1:
        parser.error("random-cases must be nonnegative and timeout-seconds positive")
    try:
        args.configs = [tuple(map(int, entry.split(":"))) for entry in args.configs.split(",")]
        if not args.configs or any(len(pair) != 2 or min(pair) < 1
                                   for pair in args.configs):
            raise ValueError
    except ValueError:
        parser.error("configs require positive feature:batch pairs")
    return args


def main():
    args = parse_args()
    # The oracle uses only pipeline metadata; native PS operations are unused.
    os.environ.setdefault("RECSTORE_DEFER_OPS_LOAD", "1")
    from recstore.ps_pipeline import PSPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("This explicit oracle requires CUDA on both ranks")
    torch.set_num_threads(1)
    device_index = args.device if args.device is not None else int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    dist.init_process_group("nccl", init_method=args.init_method,
                            timeout=timedelta(seconds=args.timeout_seconds))
    outcomes = []
    try:
        if dist.get_world_size() != 2:
            raise ValueError("The dependency-split oracle requires exactly two ranks")
        rank = dist.get_rank()
        main_thread = threading.get_ident()
        model = torch.nn.parallel.DistributedDataParallel(
            torch.nn.Linear(7, 3).to(device), device_ids=[device_index])
        original_gather = dist.all_gather
        with mock.patch.dict(os.environ, {
                "RECSTORE_PS_SEGMENT_REDUCE": "0", "RECSTORE_PS_GPU_TICKET": "0",
                "RECSTORE_PS_DEVICE_MERGE": "0", "RECSTORE_DENSE_REDUCE_OVERLAP": "0",
                "RECSTORE_PS_PREPARE_DURING_DENSE": "0"}):
            for features, maximum in args.configs:
                capacity = features * maximum
                cases = [([], []), ([0], []), ([], [0, 2]),
                         ([0, 2, 2**62], [0, 1, 2**63-1]), ([1, 1, 3], [3, 3, 4]),
                         (list(range(capacity)), list(range(capacity//2, capacity + capacity//2))),
                         ([0], [2**63-1])]
                cases = [(left[:capacity], right[:capacity]) for left, right in cases]
                generator = torch.Generator().manual_seed(args.seed + features)
                for _ in range(args.random_cases):
                    pair = []
                    for _ in range(2):
                        length = int(torch.randint(capacity + 1, (1,), generator=generator))
                        pair.append(torch.randint(0, max(1, capacity//3), (length,), generator=generator).tolist())
                    cases.append(tuple(pair))
                options = dict(table_offsets={f"f{i}": i * 100000 for i in range(features)},
                               master_table_name="unused", embedding_dim=7, device=device,
                               dependency_split=True, total_steps=len(cases) + 1,
                               max_batch_size=maximum)
                old = new = None
                try:
                    with mock.patch.dict(os.environ, RECSTORE_PS_METADATA_NCCL="0"):
                        old = PSPipeline(None, None, **options)
                    with mock.patch.dict(os.environ, RECSTORE_PS_METADATA_NCCL="1"):
                        new = PSPipeline(None, None, **options)
                    launches = []

                    def gather(outputs, source, *positional, **kwargs):
                        if kwargs.get("group") is new._metadata_group:
                            assert threading.get_ident() == main_thread
                            launches.append(source.numel())
                        return original_gather(outputs, source, *positional, **kwargs)

                    with mock.patch.object(dist, "all_gather", gather):
                        for step, pair in enumerate(cases):
                            ids = torch.tensor(pair[rank], dtype=torch.int64)
                            baseline = old._metadata_worker.submit(old._exchange_dirty_ids, step, ids)
                            if rank == step % 2 and step % 6 == 0:
                                time.sleep(0.002)
                            candidate = new._submit_nccl_dirty_ids(step, ids)
                            # Exercise a switch back to WORLD before CPU metadata consumption.
                            model.zero_grad(set_to_none=True)
                            model(torch.full((4, 7), float(rank + 1), device=device)).sum().backward()
                            for parameter in model.parameters():
                                expected = 6.0 if parameter.ndim == 2 else 4.0
                                assert torch.equal(parameter.grad, torch.full_like(parameter.grad, expected))
                            expected_ids, old_stats = baseline.result()
                            actual, new_stats = candidate.result()
                            assert torch.equal(actual, expected_ids), (features, maximum, step, rank)
                            assert actual.dtype == torch.int64 and actual.device.type == "cpu"
                            assert actual.tolist() == sorted(set(pair[0] + pair[1]))
                            for key in ("ps_pipeline_metadata_step", "ps_pipeline_local_dirty_rows",
                                        "ps_pipeline_global_dirty_rows"):
                                assert old_stats[key] == new_stats[key], key
                            assert len(new._metadata_transfers) <= 2
                            outcomes.append({"features": features, "max_batch": maximum, "step": step,
                                             "rows": actual.numel(), "union_sha256": hashlib.sha256(
                                                 actual.numpy().tobytes()).hexdigest()})
                        assert launches == [capacity + 4] * len(cases)
                    try:
                        new._adopt_cpu_ticket(0, (torch.tensor([1]),
                            torch.zeros(capacity + 1, dtype=torch.int64), capacity + 1), {}, 0.0)
                    except ValueError as error:
                        assert "max_batch_size" in str(error)
                    else:
                        raise AssertionError("raw overflow was accepted")
                finally:
                    if old is not None:
                        old.shutdown(flush=False)
                    if new is not None:
                        new.shutdown(flush=False)
                assert not new._metadata_transfers and new._metadata_group is None
            with mock.patch.dict(os.environ, RECSTORE_PS_METADATA_NCCL="1"):
                try:
                    PSPipeline(None, None, table_offsets={"a": 0}, master_table_name="unused",
                               embedding_dim=7, device=device, dependency_split=True,
                               total_steps=3, max_batch_size=9 + rank)
                except ValueError as error:
                    assert "mismatch" in str(error)
                else:
                    raise AssertionError("mismatched capacity was accepted")
        dist.barrier(device_ids=[device_index])
        result = {"status": "PASS", "rank": rank, "world": dist.get_world_size(),
                  "host": socket.gethostname(), "device": str(device), "cases": outcomes,
                  "world_ddp_backward_steps": len(outcomes), "startup_mismatch": "PASS"}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"metadata_oracle_rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "cases"}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
