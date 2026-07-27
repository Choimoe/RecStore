import os
import sys
import threading
import time
import traceback

import torch
import torch.multiprocessing as mp
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from .single_process import (
    LEARNING_RATE,
    NUM_TEST_ROUNDS,
    compare_tensors,
    get_eb_configs,
    initialize_backend_from_standard_ebc,
    to_recstore_config_dict,
)

from .config import configure_src_paths

configure_src_paths()

from recstore.KVClient import get_kv_client
from recstore.optimizer import SparseSGD
from torchrec_kv.EmbeddingBag import RecStoreEmbeddingBagCollection


BARRIER_TIMEOUT_SECONDS = 30
PROCESS_JOIN_TIMEOUT_SECONDS = 45


def ensure_spawn_start_method() -> None:
    if mp.get_start_method(allow_none=True) != "spawn":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass


def generate_rank_batch(
    num_embeddings_per_rank: int,
    batch_size: int,
    device: str,
    rank: int,
) -> KeyedJaggedTensor:
    start_key = rank * num_embeddings_per_rank
    end_key = (rank + 1) * num_embeddings_per_rank
    num_embeddings_in_range = end_key - start_key

    avg_len = max(1, (num_embeddings_in_range // batch_size) // 2)
    lengths = torch.randint(
        1,
        avg_len * 2,
        (batch_size,),
        device=device,
        dtype=torch.int32,
    )
    values = torch.randint(
        0,
        num_embeddings_in_range,
        (lengths.sum().item(),),
        device=device,
        dtype=torch.int64,
    )
    values = values + start_key

    return KeyedJaggedTensor.from_lengths_sync(
        keys=["feature_0"],
        values=values,
        lengths=lengths,
    )


def worker(rank: int, world_size: int, args, barrier, table_name: str) -> None:
    try:
        torch.manual_seed(args.seed)

        device = "cpu"
        if not args.cpu and torch.cuda.is_available():
            device = "cuda"

        num_embeddings_per_rank = args.num_embeddings
        total_embeddings = num_embeddings_per_rank * world_size
        eb_configs = get_eb_configs(
            num_embeddings=total_embeddings,
            embedding_dim=args.embedding_dim,
            table_name=table_name,
        )
        standard_ebc = EmbeddingBagCollection(tables=eb_configs, device=device)

        kv_client = get_kv_client()
        if args.ps_host and args.ps_port:
            kv_client.set_ps_config(args.ps_host, args.ps_port)

        config = eb_configs[0]
        barrier.wait(timeout=BARRIER_TIMEOUT_SECONDS)

        if rank == 0:
            initialize_backend_from_standard_ebc(kv_client, standard_ebc, config)

        barrier.wait(timeout=BARRIER_TIMEOUT_SECONDS)

        kv_client._tensor_meta[config.name] = {
            "shape": (total_embeddings, config.embedding_dim),
            "dtype": torch.float32,
        }
        kv_client._full_data_shape[config.name] = (
            total_embeddings,
            config.embedding_dim,
        )
        kv_client._data_name_list.add(config.name)
        kv_client._gdata_name_list.add(config.name)

        recstore_ebc = RecStoreEmbeddingBagCollection(
            embedding_bag_configs=to_recstore_config_dict(eb_configs),
            lr=LEARNING_RATE,
            enable_fusion=False,
        ).to(device)

        local_start = rank * num_embeddings_per_rank
        local_end = (rank + 1) * num_embeddings_per_rank
        local_keys = torch.arange(local_start, local_end, dtype=torch.int64)

        with torch.no_grad():
            pulled_weights = kv_client.pull(name=config.name, ids=local_keys)
            std_weights = standard_ebc.state_dict()[
                f"embedding_bags.{config.name}.weight"
            ][local_start:local_end].cpu()
            if not compare_tensors(std_weights, pulled_weights, f"Rank {rank} Init Sync"):
                sys.exit(1)

        standard_optimizer = torch.optim.SGD(standard_ebc.parameters(), lr=LEARNING_RATE)
        sparse_optimizer = SparseSGD([recstore_ebc], lr=LEARNING_RATE)
        torch.manual_seed(args.seed + rank + 1)

        for round_index in range(NUM_TEST_ROUNDS):
            barrier.wait(timeout=BARRIER_TIMEOUT_SECONDS)
            batch = generate_rank_batch(
                num_embeddings_per_rank,
                args.batch_size,
                device,
                rank,
            )

            standard_output = standard_ebc(batch)
            recstore_output = recstore_ebc(batch)
            if not compare_tensors(
                standard_output.values(),
                recstore_output.values(),
                f"Rank {rank} R{round_index + 1} Fwd",
            ):
                sys.exit(1)

            standard_optimizer.zero_grad()
            sparse_optimizer.zero_grad()
            standard_output.values().sum().backward()
            recstore_output.values().sum().backward()

            standard_optimizer.step()
            sparse_optimizer.step()
            sparse_optimizer.flush()

            with torch.no_grad():
                updated_std_weights = standard_ebc.state_dict()[
                    f"embedding_bags.{config.name}.weight"
                ][local_start:local_end].cpu()
                updated_rec_weights = kv_client.pull(name=config.name, ids=local_keys)
                if not compare_tensors(
                    updated_std_weights,
                    updated_rec_weights,
                    f"Rank {rank} R{round_index + 1} Wgt",
                ):
                    sys.exit(1)

        barrier.wait(timeout=BARRIER_TIMEOUT_SECONDS)
    except TimeoutError:
        print(f"[Rank {rank}] Timed out waiting on multiprocessing barrier")
        traceback.print_exc()
        sys.exit(2)
    except threading.BrokenBarrierError:
        print(f"[Rank {rank}] Barrier broken because another rank failed")
        traceback.print_exc()
        sys.exit(3)
    except Exception:
        print(f"[Rank {rank}] Worker failed unexpectedly")
        traceback.print_exc()
        sys.exit(4)


def run_multiprocess_precision(args) -> None:
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ensure_spawn_start_method()
    world_size = args.world_size
    barrier = mp.Barrier(world_size)
    processes = []
    table_name = f"table_mp_{int(time.time())}_{os.getpid()}"

    for rank in range(world_size):
        process = mp.Process(
            target=worker,
            args=(rank, world_size, args, barrier, table_name),
        )
        process.start()
        processes.append(process)

    failed = False
    for process in processes:
        process.join(PROCESS_JOIN_TIMEOUT_SECONDS)
        if process.is_alive():
            print(
                f"Worker PID {process.pid} did not finish within "
                f"{PROCESS_JOIN_TIMEOUT_SECONDS}s; terminating."
            )
            process.terminate()
            process.join(5)
            failed = True
        if process.exitcode != 0:
            failed = True

    if failed:
        sys.exit(1)

    print("Multiprocess precision test completed successfully.")

