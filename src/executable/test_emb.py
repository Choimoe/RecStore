import random
import numpy as np
import datetime
import argparse
import debugpy
import tqdm
import pytest
import os
import time
from collections import namedtuple
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


import recstore
from recstore import KGCacheControllerWrapperBase
from recstore.cache import CacheEmbFactory, TorchNativeStdEmbDDP
from recstore.PsKvstore import ShmKVStore, kvinit
from recstore import DistEmbedding, BasePerfSampler, TestPerfSampler, Mfence
from recstore.utils import XLOG, Timer, GPUTimer, xmh_nvtx_range
import recstore.DistOpt as DistOpt


random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)


EmbContext = namedtuple('EmbContext', ["emb_name", 'sparse_opt', 'dist_opt'])


USE_SGD = True
# USE_SGD = False
LR = 2

# XMH_DEBUG = True
XMH_DEBUG = False

CHECK = True
# CHECK = False


if XMH_DEBUG:
    logging.basicConfig(format='%(levelname)-2s [%(process)d %(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m-%d:%H:%M:%S', level=logging.DEBUG)
else:
    logging.basicConfig(format='%(levelname)-2s [%(process)d %(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m-%d:%H:%M:%S', level=logging.INFO)


def worker_main(routine, worker_id, num_workers, emb_context, args):
    torch.cuda.set_device(worker_id)
    torch.manual_seed(worker_id)
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12545')
    world_size = num_workers
    torch.distributed.init_process_group(backend=None,
                                         init_method=dist_init_method,
                                         world_size=world_size,
                                         rank=worker_id,
                                         timeout=datetime.timedelta(seconds=100))
    routine(worker_id, num_workers, emb_context, args)


class TestShardedCache:

    if XMH_DEBUG:
        num_workers = 4
        EMB_DIM = 3
        # EMB_LEN = 100
        EMB_LEN = 2000000
        BATCH_SIZE = 1024
    else:
        num_workers = 4
        # EMB_DIM = 32
        EMB_DIM = 3
        EMB_LEN = 2000000
        BATCH_SIZE = 1024

    def main_routine(self, routine, args=None):
        # wrap rountine with dist_init
        print(
            f"========== Running Test with routine {routine} {args}==========")

        kvinit()
        emb = DistEmbedding(TestShardedCache.EMB_LEN,
                            TestShardedCache.EMB_DIM, name="full_emb",)

        fake_tensor = torch.Tensor([0])
        if USE_SGD:
            sparse_opt = optim.SGD(
                [fake_tensor], lr=LR,)
            dist_opt = DistOpt.SparseSGD(
                [emb], lr=LR)
        else:
            sparse_opt = optim.Adam([fake_tensor], lr=LR)
            dist_opt = DistOpt.SparseAdagrad(
                [emb], lr=LR)

        # import copy
        # deep_copy_dist_opt = copy.deepcopy(dist_opt)
        emb_context = EmbContext(
            emb_name=emb.name, sparse_opt=sparse_opt, dist_opt=None)

        XLOG.cdebug(
            f"ShmKVStore.tensor_store {hex(ShmKVStore.tensor_store['full_emb'].data_ptr())}")

        workers = []
        for worker_id in range(0, TestShardedCache.num_workers):
            p = mp.Process(target=worker_main, args=(
                routine, worker_id, TestShardedCache.num_workers, emb_context, args))
            p.start()
            XLOG.info(f"Worker {worker_id} pid={p.pid}")
            workers.append(p)
        # worker_main(
        #     routine, 0, TestShardedCache.num_workers, emb_context, args)

        for each in workers:
            each.join()
            # assert each.exitcode == 0

        print("join all processes done")

    def init_emb_tensor(self, emb, rank, num_workers):
        dist.barrier()
        XLOG.info(f"emb.data_ptr={hex(emb.get_shm_tensor().data_ptr())}")
        linspace = np.linspace(0, emb.shape[0], num_workers+1, dtype=int)

        assert rank == dist.get_rank()
        if rank == 0:
            print(f"rank {rank} start initing emb")
            for i in tqdm.trange(linspace[rank], linspace[rank + 1]):
                emb.weight[i] = torch.ones(emb.shape[1]) * i
        else:
            for i in range(linspace[rank], linspace[rank + 1]):
                emb.weight[i] = torch.ones(emb.shape[1]) * i
        dist.barrier()
        Mfence.mfence()
        for i in range(1000):
            idx = random.randint(0, emb.shape[0]-1)
            # idx = i
            if not (torch.allclose(
                    emb.weight[idx], torch.ones(emb.shape[1]) * idx)):
                XLOG.error(
                    f"init failed, idx={idx}, emb[idx]={emb.weight[idx]}")
                assert False
        dist.barrier()

    def routine_cache_helper(self, worker_id, num_workers, emb_context, args):
        rank = dist.get_rank()
        XLOG.debug(f"rank{rank}: pid={os.getpid()}")
        kvinit()
        emb = DistEmbedding(TestShardedCache.EMB_LEN,
                            TestShardedCache.EMB_DIM, name=emb_context.emb_name)
        dist.barrier()

        sparse_opt = emb_context.sparse_opt
        # dist_opt = emb_context.dist_opt

        dist_opt = DistOpt.SparseSGD(
            [emb], lr=LR)

        XLOG.debug(f'dist_opt._params = {dist_opt._params}')

        XLOG.debug(
            f"in rank{rank}, full_emb.data_ptr={hex(emb.get_shm_tensor().data_ptr())}")

        self.init_emb_tensor(emb, worker_id, num_workers)

        args['num_workers'] = num_workers
        args['num_gpus'] = num_workers
        args['clr'] = dist_opt.lr
        args['full_emb_capacity'] = emb.shape[0]
        json_str = json.dumps(args)

        if rank == 0:
            print("------------json------------")
            print(json_str)

        # Generate standard embedding
        print("TorchNativeStdEmbDDP(emb, device='cuda')", flush=True)
        std_emb = TorchNativeStdEmbDDP(emb, device='cuda')
        # std_emb = TorchNativeStdEmbDDP(emb, device='cpu')

        std_emb.reg_opt(sparse_opt)
        print("TorchNativeStdEmbDDP(emb, device='cuda') done", flush=True)
        # Generate standard embedding done

        print("CacheEmbFactory.New(cached_emb_type, emb, args)", flush=True)
        # Generate our embedding
        cached_emb_type = args['test_cache']
        abs_emb = CacheEmbFactory.New(cached_emb_type, emb, args)
        abs_emb.reg_opt(sparse_opt)
        print("CacheEmbFactory.New(cached_emb_type, emb, args) done", flush=True)

        print("TestPerfSampler", flush=True)
        test_perf_sampler = TestPerfSampler(rank=rank,
                                            L=args['L'],
                                            num_ids_per_step=TestShardedCache.BATCH_SIZE,
                                            full_emb_capacity=emb.shape[0],
                                            backmode=args['backwardMode'],
                                            )
        print("TestPerfSampler done", flush=True)

        print("kg_cache_controller = KGCacheControllerWrapper")
        if cached_emb_type == "KnownLocalCachedEmbedding":
            kg_cache_controller = KGCacheControllerWrapperBase.FactoryNew(
                "RecStore", json_str)

        else:
            kg_cache_controller = KGCacheControllerWrapperBase.FactoryNew(
                "Dummy", "")

        print("kg_cache_controller.init")
        kg_cache_controller.init()
        test_perf_sampler.Prefill()
        print("test_perf_sampler.Prefill")

        # Generate our embedding done
        timer_Forward = Timer("Forward")
        timer_Backward = Timer("Backward")
        timer_Optimize = Timer("Optimize")

        timer_start = Timer(f"E2E-{args['log_interval']}")
        timer_start.start()
        # forward
        for _ in tqdm.trange(900):
            sparse_opt.zero_grad()
            dist_opt.zero_grad()

            print(f"========== Step {_} ========== ", flush=True)

            if _ % args['log_interval'] == (args['log_interval']-1):
                timer_start.stop()
                timer_start.start()

            input_keys = next(test_perf_sampler)

            # torch.set_printoptions(profile="full")
            XLOG.cdebug(f"{rank}:step{_}, input_keys {input_keys}")
            # torch.set_printoptions(profile="default")

            if CHECK:
                std_embed_value = std_emb.forward(input_keys).cuda()
                std_loss = std_embed_value.sum()
                print(std_loss, std_loss.shape)
                std_loss.backward()
                XLOG.cdebug(
                    f"{rank}:std_embed_value {std_embed_value}")

            timer_Forward.start()
            embed_value = abs_emb.forward(input_keys).cuda()
            timer_Forward.stop()
            XLOG.cdebug(f"{rank}:embed_value {embed_value}")
            # loss = embed_value.sum(-1).sum(-1)
            loss = embed_value.sum()
            timer_Backward.start()
            loss.backward()
            timer_Backward.stop()

            if CHECK:
                if not torch.allclose(
                        embed_value, std_embed_value):
                    for i in range(len(input_keys)):
                        if not torch.allclose(
                                embed_value[i], std_embed_value[i]):
                            XLOG.error(
                                f"rank{rank}: forward failed, input_key={input_keys[i]}, \
                                    embed_value={embed_value[i]}, \
                                    std_embed_value={std_embed_value[i]}")
                            kg_cache_controller.StopThreads()

                            assert False, "forward is error"
                    assert False, "forward is error"

                assert (torch.allclose(
                    loss, std_loss))

            timer_Optimize.start()
            sparse_opt.step()
            dist_opt.step()
            timer_Optimize.stop()

            kg_cache_controller.AfterBackward()
            # if rank == 0:
            #     XLOG.info(f"rank{rank}: step{_} done")
            #     kg_cache_controller.controller.PrintPq()

        if rank == 0:
            kg_cache_controller.GraceFullyStopThreads()

    def test_known_sharded_cache(self,):
        # for test_cache in ["KnownShardedCachedEmbedding", "KnownLocalCachedEmbedding"]:

        config = {
            # "test_cache": ['TorchNativeStdEmb', ],
            "test_cache": ['KnownLocalCachedEmbedding', ],

            # "test_cache": ['KnownLocalCachedEmbedding',
            #                     'KnownShardedCachedEmbedding',
            #                     'TorchNativeStdEmb'],

            # "backwardMode": ["PySync", "CppSync", "CppAsync", "CppAsyncV2"],

            # "backwardMode": ["PySync",],
            # "backwardMode": ["CppSync",],
            # "backwardMode": ["CppAsync",],
            "backwardMode": ["CppAsyncV2",],

            # 0没问题，1、2的时候会有bug，应该也是多线程的Upsert时候的问题，但目前来看不影响性能
            "update_pq_use_omp": [1],

            "kUseParallelClean": [1],
            # "kUseParallelClean": [0],

            # V1: 11 OK
            # V2: 00 OK  01 OK  #10 OK 11错的  20错的  21错的

            # "backgrad_init":['cpu', 'gpu', 'both'],
            "backgrad_init": ['both'],

            # "cache_ratio": [0.1, 0.3, 0.5],
            "cache_ratio": [0.1,],
        }

        def GenProduct(config):
            import itertools
            keys, values = zip(*config.items())
            permutations_config = [dict(zip(keys, v))
                                   for v in itertools.product(*values)]
            return permutations_config

        for each in GenProduct(config):
            test_cache = each['test_cache']
            backmode = each['backwardMode']
            cache_ratio = each['cache_ratio']
            backgrad_init = each['backgrad_init']
            update_pq_use_omp = each['update_pq_use_omp']

            KGCacheControllerWrapperBase.BeforeDDPInit()
            args = {
                "kForwardItersPerStep": 1,
                "L": 10,
                "nr_background_threads": 4,
                "log_interval": 100,
                **each,
            }
            print("xmh: ", args)
            self.main_routine(self.routine_cache_helper, args)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    test = TestShardedCache()
    test.test_known_sharded_cache()
