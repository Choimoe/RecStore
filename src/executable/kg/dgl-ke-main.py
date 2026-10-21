import logging

logging.basicConfig(
    format="%(levelname)-2s [%(process)d %(filename)s:%(lineno)d %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


from dglke.dataloader import KGDataset, TrainDataset, NewBidirectionalOneShotIterator

# import test_utils
import pickle
from dglke.utils import get_compatible_batch_size, save_model, CommonArgParser
from dglke.dataloader import get_dataset
from dglke.dataloader import (
    ConstructGraph,
    EvalDataset,
    TrainDataset,
    NewBidirectionalOneShotIterator,
)
import dglke
import os
import gc
import time
import json
import os
import datetime

import dgl
import random
import torch
import torch as th
import numpy as np

import recstore
from recstore.cache import CacheShardingPolicy  # nopep8

from recstore import GraphCachedSampler, KGCacheControllerWrapperBase

random.seed(0)
np.random.seed(0)
# torch.use_deterministic_algorithms(True)


backend = os.environ.get("DGLBACKEND", "pytorch")
if backend.lower() == "mxnet":
    import multiprocessing as mp
    from dglke.train_mxnet import load_model
    from dglke.train_mxnet import train
    from dglke.train_mxnet import test
else:
    import torch.multiprocessing as mp
    from dglke.train_pytorch import load_model
    from dglke.train_pytorch import train, train_mp
    from dglke.train_pytorch import test, test_mp


def CreateSamplers(args, kg_dataset: KGDataset, train_data: TrainDataset):
    return train_data.CreateSamplers(
        num_nodes=kg_dataset.n_entities,
        is_chunked=True,
        batch_size=args.batch_size,
        neg_sample_size=args.neg_sample_size,
        neg_chunk_size=args.neg_sample_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        exclude_positive=False,
        has_edge_importance=args.has_edge_importance,
        renumbering_dict=None,
        real_train=True,
    )


class ArgParser(CommonArgParser):
    def list_of_ints(arg):
        return list(map(int, arg.split(",")))

    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ["true", "false"]:
            raise ValueError("Need bool; got %r" % s)
        return {"true": True, "false": False}[s.lower()]

    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument("--nr_gpus", type=int, default=-1, help="# of gpus")
        self.add_argument(
            "--gpu",
            type=ArgParser.list_of_ints,
            default=[-1],
            help="A list of gpu ids, e.g. 0,1,2,4",
        )
        self.add_argument(
            "--mix_cpu_gpu",
            type=bool,
            help="Training a knowledge graph embedding model with both CPUs and GPUs."
            "The embeddings are stored in CPU memory and the training is performed in GPUs."
            "This is usually used for training a large knowledge graph embeddings.",
        )
        self.add_argument(
            "--valid",
            action="store_true",
            help="Evaluate the model on the validation set in the training.",
        )
        self.add_argument(
            "--rel_part",
            action="store_true",
            help="Enable relation partitioning for multi-GPU training.",
        )
        self.add_argument(
            "--async_update",
            type=bool,
            help="Allow asynchronous update on node embedding for multi-GPU training."
            "This overlaps CPU and GPU computation to speed up.",
        )
        self.add_argument(
            "--has_edge_importance",
            action="store_true",
            help="Allow providing edge importance score for each edge during training."
            "The positive score will be adjusted "
            "as pos_score = pos_score * edge_importance",
        )
        self.add_argument("--cached_emb_type", type=str, help=".")
        self.add_argument(
            "--use_my_emb", type=ArgParser._str_to_bool, required=True, help="."
        )
        self.add_argument("--cache_ratio", type=float, required=True, help=".")
        self.add_argument("--shuffle", type=bool, default=False, help=".")
        self.add_argument(
            "--backwardMode",
            type=str,
            required=True,
            choices=["PySync", "CppSync", "CppAsync", "CppAsyncV2"],
            help=".",
        )
        self.add_argument("--L", type=int, default=10, help="lookahead value")
        self.add_argument(
            "--nr_background_threads", type=int, default=32, help="flush threads"
        )
        self.add_argument("--update_cache_use_omp", type=int, default=1, help="use omp")
        self.add_argument("--update_pq_use_omp", type=int, default=2, help="use omp")


def prepare_save_path(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    folder = "{}_{}_".format(args.model_name, args.dataset)
    n = len([x for x in os.listdir(args.save_path) if x.startswith(folder)])
    folder += str(n)
    args.save_path = os.path.join(args.save_path, folder)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


def main():
    args = ArgParser().parse_args()
    if args.use_my_emb == False:
        # write a environment variable to indicate a small IPC memory
        os.environ["SHM_GB"] = "10"
    else:
        os.environ["SHM_GB"] = ""
        

    json_str = f"""{{
        "num_gpus": {args.nr_gpus},
        "L": {args.L},
        "backgrad_init": "both", 
        "kForwardItersPerStep": 2,
        "clr": 0.01,
        "nr_background_threads": {args.nr_background_threads},
        "backwardMode": "{args.backwardMode}",
        "cache_ratio": {args.cache_ratio},
        "update_cache_use_omp":  {args.update_cache_use_omp},
        "update_pq_use_omp":  {args.update_pq_use_omp}
        }}"""

    KGCacheControllerWrapperBase.BeforeDDPInit()

    json_config = json.loads(json_str)

    args.kForwardItersPerStep = json_config["kForwardItersPerStep"]
    args.backgrad_init = json_config["backgrad_init"]

    from recstore.PsKvstore import ShmKVStore, kvinit

    kvinit()

    recstore.IPCTensorFactory.ClearIPCMemory()

    if args.nr_gpus == 0:
        args.gpu = [-1]
    else:
        args.gpu = list(range(args.nr_gpus))

    prepare_save_path(args)

    init_time_start = time.time()
    # load dataset and samplers
    logging.warning(f"get_dataset, {args.data_path}")
    dataset = get_dataset(
        args.data_path,
        args.dataset,
        args.format,
        args.delimiter,
        args.data_files,
        args.has_edge_importance,
    )
    logging.warning(f"get_dataset done, {args.data_path}")

    if args.neg_sample_size_eval < 0:
        args.neg_sample_size_eval = dataset.n_entities
    args.batch_size = get_compatible_batch_size(args.batch_size, args.neg_sample_size)
    args.batch_size_eval = get_compatible_batch_size(
        args.batch_size_eval, args.neg_sample_size_eval
    )
    # We should turn on mix CPU-GPU training for multi-GPU training.

    args.mix_cpu_gpu = True

    if len(args.gpu) > 1:
        if args.num_proc < len(args.gpu):
            args.num_proc = len(args.gpu)
    # We need to ensure that the number of processes should match the number of GPUs.
    if len(args.gpu) > 1 and args.num_proc > 1:
        assert (
            args.num_proc % len(args.gpu) == 0
        ), "The number of processes needs to be divisible by the number of GPUs"
    # For multiprocessing training, we need to ensure that training processes are synchronized periodically.
    if args.num_proc > 1:
        # args.force_sync_interval = 1000
        args.force_sync_interval = 1

    args.eval_filter = not args.no_eval_filter
    if args.neg_deg_sample_eval:
        assert (
            not args.eval_filter
        ), "if negative sampling based on degree, we can't filter positive edges."

    args.soft_rel_part = args.mix_cpu_gpu and args.rel_part

    print("ARGS: ", args)

    logging.warning(f"ConstructGraph")
    g = ConstructGraph(dataset, args)
    logging.warning(f"ConstructGraph done")

    print("xmh xmh ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
    print(g)

    logging.warning(f"Construct TrainDataset")
    train_data = TrainDataset(
        g, dataset, args, ranks=args.num_proc, has_importance=args.has_edge_importance
    )
    logging.warning(f"Construct TrainDataset done")
    # if there is no cross partition relaiton, we fall back to strict_rel_part
    args.strict_rel_part = args.mix_cpu_gpu and (train_data.cross_part == False)
    args.num_workers = 8  # fix num_worker to 8

    # import pyinstrument

    # profiler = pyinstrument.Profiler()
    # profiler.start()

    logging.warning(f"CreateSamplers")
    train_samplers = CreateSamplers(args, kg_dataset=dataset, train_data=train_data)
    logging.warning(f"CreateSamplers done")


    logging.warning(f"train_data.PreSampling")
    renumbering_dict, cache_sizes_all_rank = train_data.PreSampling(
        train_samplers,
        args.batch_size,
        args.cache_ratio,
        args.neg_sample_size,
        args.neg_sample_size,
        #   args.num_workers,
        num_workers=8,
        shuffle=args.shuffle,
        exclude_positive=False,
        has_edge_importance=False,
    )
    logging.warning(f"train_data.PreSampling done")

    logging.warning(f"CacheShardingPolicy.set_presampling")
    CacheShardingPolicy.set_presampling(cache_sizes_all_rank)
    logging.warning(f"train_data.RenumberingGraph")
    train_data.RenumberingGraph(renumbering_dict)
    logging.warning(f"train_data.RenumberingGraph done")

    logging.warning(f"BatchCreateSamplers")
    train_samplers = GraphCachedSampler.BatchCreateCachedSamplers(
        args.L, train_samplers, backmode=json_config["backwardMode"]
    )
    logging.warning(f"BatchCreateSamplers done")

    # grad_clients = controller.CreateGradClients()
    grad_clients = [None for _ in range(args.num_proc)]

    rel_parts = (
        train_data.rel_parts if args.strict_rel_part or args.soft_rel_part else None
    )
    cross_rels = train_data.cross_rels if args.soft_rel_part else None
    train_data = None
    gc.collect()

    if args.valid or args.test:
        if len(args.gpu) > 1:
            args.num_test_proc = (
                args.num_proc if args.num_proc < len(args.gpu) else len(args.gpu)
            )
        else:
            args.num_test_proc = args.num_proc
        if args.valid:
            assert dataset.valid is not None, "validation set is not provided"
        if args.test:
            assert dataset.test is not None, "test set is not provided"
        eval_dataset = EvalDataset(g, dataset, args)

    if args.valid:
        if args.num_proc > 1:
            valid_sampler_heads = []
            valid_sampler_tails = []
            if args.dataset == "wikikg90M":
                for i in range(args.num_proc):
                    valid_sampler_tail = eval_dataset.create_sampler_wikikg90M(
                        "valid",
                        args.batch_size_eval,
                        mode="tail",
                        rank=i,
                        ranks=args.num_proc,
                    )
                    valid_sampler_tails.append(valid_sampler_tail)
            else:
                for i in range(args.num_proc):
                    valid_sampler_head = eval_dataset.create_sampler(
                        "valid",
                        args.batch_size_eval,
                        args.neg_sample_size_eval,
                        args.neg_sample_size_eval,
                        args.eval_filter,
                        mode="chunk-head",
                        num_workers=args.num_workers,
                        rank=i,
                        ranks=args.num_proc,
                    )
                    valid_sampler_tail = eval_dataset.create_sampler(
                        "valid",
                        args.batch_size_eval,
                        args.neg_sample_size_eval,
                        args.neg_sample_size_eval,
                        args.eval_filter,
                        mode="chunk-tail",
                        num_workers=args.num_workers,
                        rank=i,
                        ranks=args.num_proc,
                    )
                    valid_sampler_heads.append(valid_sampler_head)
                    valid_sampler_tails.append(valid_sampler_tail)
        else:  # This is used for debug
            if args.dataset == "wikikg90M":
                valid_sampler_tail = eval_dataset.create_sampler_wikikg90M(
                    "valid", args.batch_size_eval, mode="tail", rank=0, ranks=1
                )
            else:
                valid_sampler_head = eval_dataset.create_sampler(
                    "valid",
                    args.batch_size_eval,
                    args.neg_sample_size_eval,
                    args.neg_sample_size_eval,
                    args.eval_filter,
                    mode="chunk-head",
                    num_workers=args.num_workers,
                    rank=0,
                    ranks=1,
                )
                valid_sampler_tail = eval_dataset.create_sampler(
                    "valid",
                    args.batch_size_eval,
                    args.neg_sample_size_eval,
                    args.neg_sample_size_eval,
                    args.eval_filter,
                    mode="chunk-tail",
                    num_workers=args.num_workers,
                    rank=0,
                    ranks=1,
                )
    if args.test:
        if args.num_test_proc > 1:
            test_sampler_tails = []
            test_sampler_heads = []
            if args.dataset == "wikikg90M":
                for i in range(args.num_proc):
                    valid_sampler_tail = eval_dataset.create_sampler_wikikg90M(
                        "test",
                        args.batch_size_eval,
                        mode="tail",
                        rank=i,
                        ranks=args.num_proc,
                    )
                    valid_sampler_tails.append(valid_sampler_tail)
            else:
                for i in range(args.num_test_proc):
                    test_sampler_head = eval_dataset.create_sampler(
                        "test",
                        args.batch_size_eval,
                        args.neg_sample_size_eval,
                        args.neg_sample_size_eval,
                        args.eval_filter,
                        mode="chunk-head",
                        num_workers=args.num_workers,
                        rank=i,
                        ranks=args.num_test_proc,
                    )
                    test_sampler_tail = eval_dataset.create_sampler(
                        "test",
                        args.batch_size_eval,
                        args.neg_sample_size_eval,
                        args.neg_sample_size_eval,
                        args.eval_filter,
                        mode="chunk-tail",
                        num_workers=args.num_workers,
                        rank=i,
                        ranks=args.num_test_proc,
                    )
                    test_sampler_heads.append(test_sampler_head)
                    test_sampler_tails.append(test_sampler_tail)
        else:
            if args.dataset == "wikikg90M":
                test_sampler_tail = eval_dataset.create_sampler_wikikg90M(
                    "test", args.batch_size_eval, mode="tail", rank=0, ranks=1
                )
            else:
                test_sampler_head = eval_dataset.create_sampler(
                    "test",
                    args.batch_size_eval,
                    args.neg_sample_size_eval,
                    args.neg_sample_size_eval,
                    args.eval_filter,
                    mode="chunk-head",
                    num_workers=args.num_workers,
                    rank=0,
                    ranks=1,
                )
                test_sampler_tail = eval_dataset.create_sampler(
                    "test",
                    args.batch_size_eval,
                    args.neg_sample_size_eval,
                    args.neg_sample_size_eval,
                    args.eval_filter,
                    mode="chunk-tail",
                    num_workers=args.num_workers,
                    rank=0,
                    ranks=1,
                )

    # load model
    n_entities = dataset.n_entities
    n_relations = dataset.n_relations
    emap_file = dataset.emap_fname
    rmap_file = dataset.rmap_fname

    # We need to free all memory referenced by dataset.
    eval_dataset = None
    dataset = None
    gc.collect()

    logging.warning(f"Start load_model")
    model = load_model(args, n_entities, n_relations)
    logging.warning(f"Stop load_model")

    logging.warning(f"Start model.share_memory")
    if args.num_proc > 1 or args.async_update:
        model.share_memory()
    logging.warning(f"Stop model.share_memory")

    logging.warning(
        "Total initialize time {:.3f} seconds".format(time.time() - init_time_start)
    )

    # train
    if args.num_proc > 1:
        procs = []
        barrier = mp.Barrier(args.num_proc)
        for i in range(1, args.num_proc):
            if args.dataset == "wikikg90M":
                valid_sampler = [valid_sampler_tails[i]] if args.valid else None
            else:
                valid_sampler = (
                    [valid_sampler_heads[i], valid_sampler_tails[i]]
                    if args.valid
                    else None
                )
            proc = mp.Process(
                target=train_mp,
                args=(
                    json_str,
                    args,
                    model,
                    train_samplers[i],
                    valid_sampler,
                    i,
                    rel_parts,
                    cross_rels,
                    barrier,
                    grad_clients[i],
                ),
            )
            procs.append(proc)
            proc.start()
            print(f"[Rank{i}] pid = {proc.pid}")

        train_mp(
            json_str,
            args,
            model,
            train_samplers[0],
            None,
            0,
            rel_parts,
            cross_rels,
            barrier,
            grad_clients[i],
        )

        for i, proc in enumerate(procs):
            proc.join()
            assert proc.exitcode == 0
    else:
        if args.dataset == "wikikg90M":
            valid_samplers = [valid_sampler_tail] if args.valid else None
        else:
            valid_samplers = (
                [valid_sampler_head, valid_sampler_tail] if args.valid else None
            )
        train(args, model, train_samplers[0], valid_samplers, rel_parts=rel_parts)

    if not args.no_save_emb:
        save_model(args, model, emap_file, rmap_file)

    # test
    if args.test:
        start = time.time()
        if args.num_test_proc > 1:
            queue = mp.Queue(args.num_test_proc)
            procs = []
            for i in range(args.num_test_proc):
                if args.dataset == "wikikg90M":
                    proc = mp.Process(
                        target=test_mp,
                        args=(args, model, [test_sampler_tails[i]], i, "Test", queue),
                    )
                else:
                    proc = mp.Process(
                        target=test_mp,
                        args=(
                            args,
                            model,
                            [test_sampler_heads[i], test_sampler_tails[i]],
                            i,
                            "Test",
                            queue,
                        ),
                    )
                procs.append(proc)
                proc.start()

            if args.dataset == "wikikg90M":
                print("The predict results have saved to {}".format(args.save_path))
            else:
                total_metrics = {}
                metrics = {}
                logs = []
                for i in range(args.num_test_proc):
                    log = queue.get()
                    logs = logs + log

                for metric in logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
                print("-------------- Test result --------------")
                for k, v in metrics.items():
                    print("Test average {} : {}".format(k, v))
                print("-----------------------------------------")

            for proc in procs:
                proc.join()
                assert proc.exitcode == 0
        else:
            if args.dataset == "wikikg90M":
                test(args, model, [test_sampler_tail])
            else:
                test(args, model, [test_sampler_head, test_sampler_tail])
            if args.dataset == "wikikg90M":
                print("The predict results have saved to {}".format(args.save_path))
        print("testing takes {:.3f} seconds".format(time.time() - start))


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(5678)
    # print("wait debugpy connect", flush=True)
    # debugpy.wait_for_client()
    main()
