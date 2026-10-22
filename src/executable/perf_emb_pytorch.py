from contextlib import contextmanager
from pyinstrument import Profiler
import numpy as np
import unittest
import datetime
import time
import argparse
import debugpy
import json
import tqdm
import time
import logging

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function, ProfilerActivity

import recstore
from recstore.PsKvstore import kvinit
from recstore import DistEmbedding, BasePerfSampler
from recstore.utils import XLOG, Timer, GPUTimer, xmh_nvtx_range
import recstore.DistOpt as DistOpt


import random
random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)


LR = 1
# DIFF_TEST = True
DIFF_TEST = False


class PerfSampler(BasePerfSampler):
    def __init__(self, rank, L, num_ids_per_step, full_emb_capacity, backmode,
                 distribution,
                 alpha,
                 ) -> None:
        super().__init__(rank, L, num_ids_per_step, backmode)

        if distribution == 'uniform':
            pass
        elif distribution == 'zipf':
            self.zipfianTorchFiller = recstore.ZipfianTorchFiller(
                full_emb_capacity, alpha)

        self.full_emb_capacity = full_emb_capacity
        self.distribution = distribution
        self.rank = rank

    def gen_next_sample(self):
        if self.distribution == 'uniform':
            return self.UniformGen()
        else:
            return self.ZipfianGen()

    def UniformGen(self):
        # print(f'rank{self.rank} reached before next sampler {time.time()}')
        entity_id = th.randint(self.full_emb_capacity, size=(
            self.num_ids_per_step,), dtype=th.int64).cuda()
        # print(f'rank{self.rank} reached after next sampler {time.time()}')
        return entity_id

    def ZipfianGen(self):
        entity_id = th.empty((self.num_ids_per_step,), dtype=th.int64)
        self.zipfianTorchFiller.fillArrayTorch(entity_id)
        return entity_id.cuda()


def get_run_config():
    def parse_args(default_run_config):
        argparser = argparse.ArgumentParser("Training")
        argparser.add_argument('--num_workers', type=int,
                               default=4)
        argparser.add_argument('--num_embs', type=int,
                               default=int(10*1e6))
        #    default=1*1e6)
        argparser.add_argument('--emb_dim', type=int,
                               default=32)
        argparser.add_argument('--L', type=int,
                               default=10)
        argparser.add_argument('--with_perf', type=bool,
                               default=False)
        argparser.add_argument('--batch_size', type=int,
                               default=5000)
        argparser.add_argument('--cache_ratio', type=float,
                               default=0.1)
        argparser.add_argument('--log_interval', type=int,
                               default=50)
        argparser.add_argument('--run_steps', type=int,
                               default=500)
        argparser.add_argument('--backwardMode', choices=["CppSync",
                                                          "CppAsync",
                                                          "CppAsyncV2",
                                                          "PySync"],
                               default="PySync"
                               )
        argparser.add_argument('--backgrad_init', choices=["cpu",
                                                           "gpu",
                                                           "both",
                                                           ],
                               default="both"
                               )
        argparser.add_argument('--kForwardItersPerStep', type=int,
                               default=1)
        argparser.add_argument('--nr_background_threads', type=int,
                               default=16)

        argparser.add_argument('--distribution', choices=['uniform', 'zipf'],
                               default='zipf')
        argparser.add_argument('--zipf_alpha', type=float, default=0.99)

        return vars(argparser.parse_args())

    run_config = {}
    run_config.update(parse_args(run_config))
    return run_config


def init_emb_tensor(emb, worker_id, num_workers):
    if not DIFF_TEST:
        return
    dist.barrier()
    linspace = np.linspace(0, emb.shape[0], num_workers+1, dtype=int)
    if worker_id == 0:
        print(f"rank {worker_id} start initing emb")
        for i in tqdm.trange(linspace[worker_id], linspace[worker_id + 1]):
            emb.weight[i] = torch.ones(emb.shape[1]) * i
    else:
        for i in range(linspace[worker_id], linspace[worker_id + 1]):
            emb.weight[i] = torch.ones(emb.shape[1]) * i
    dist.barrier()
    Mfence.mfence()
    for i in range(1000):
        idx = random.randint(0, emb.shape[0]-1)
        if not (torch.allclose(
                emb.weight[idx], torch.ones(emb.shape[1]) * idx)):
            XLOG.error(
                f"init failed, idx={idx}, emb[idx]={emb.weight[idx]}")
            assert False
    dist.barrier()


def main_routine(args, routine):
    # wrap rountine with dist_init
    def worker_main(routine, worker_id, args):
        torch.cuda.set_device(worker_id)
        torch.manual_seed(worker_id)
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12335')
        world_size = args['num_workers']
        torch.distributed.init_process_group(backend=None,
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=worker_id,
                                             timeout=datetime.timedelta(seconds=100))
        routine(worker_id, args)

    kvinit()

    # !!!!!!!!!!Don't init dist embedding in the main process, init it after fork !!!!!!!!!!!
    # emb = DistEmbedding(int(args['num_embs']),
    #                     int(args['emb_dim']), name="full_emb",)
    # XLOG.warn("After init DistEmbedding")
    # # dummy LR, only register the tensor state of OSP
    # dist_opt = DistOpt.SparseSGD([emb], lr=LR)

    print(f"========== Running Perf with routine {routine}==========")
    workers = []
    for worker_id in range(1, args['num_workers']):
        p = mp.Process(target=worker_main, args=(
            routine, worker_id, args))
        p.start()
        print(f"Worker {worker_id} pid={p.pid}")
        workers.append(p)

    worker_main(routine, 0, args)

    for each in workers:
        each.join()
        assert each.exitcode == 0


def routine_local_cache_helper(worker_id, args):
    USE_SGD = True
    # USE_SGD = False
    rank = dist.get_rank()
    emb = DistEmbedding(int(args['num_embs']),
                        int(args['emb_dim']), name="full_emb",)
    XLOG.debug(
        f"in rank{rank}, full_emb.data_ptr={hex(emb.get_shm_tensor().data_ptr())}")
    dist.barrier()
    # print("begin full_emb.get_shm_tensor().zero_()", flush=True)
    # emb.get_shm_tensor().zero_()
    # print("end full_emb.get_shm_tensor().zero_()",  flush=True)

    init_emb_tensor(emb, worker_id, args['num_workers'])
    # Generate our embedding
    fake_tensor = torch.Tensor([0])
    sparse_opt = optim.SGD(
        [fake_tensor], lr=LR,)
    dist_opt = DistOpt.SparseSGD([emb], lr=LR)
    abs_emb = CacheEmbFactory.New(args["emb_choice"], emb, args)
    abs_emb.reg_opt(sparse_opt)
    dist.barrier()
    # Generate our embedding done

    if DIFF_TEST:
        # Generate standard embedding
        std_emb = TorchNativeStdEmbDDP(emb, device='cuda')
        std_emb.reg_opt(sparse_opt)
        # Generate standard embedding done

    # forward
    start = time.time()
    start_step = 0
    warmup_iters = 50
    # warmup_iters = 0

    if args['with_perf']:
        for_range = tqdm.trange(args['run_steps'])
        with_perf = True
    else:
        for_range = range(args['run_steps'])
        with_perf = False

    with_pyinstrucment = False
    # with_pyinstrucment = True
    with_cudaPerf = False
    with_torchPerf = True

    if with_perf and with_pyinstrucment:
        pyinstruct_profiler = Profiler()

    if with_perf and with_torchPerf:
        torch_profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)

    perf_sampler = PerfSampler(rank=rank,
                               L=args['L'],
                               num_ids_per_step=args['batch_size'],
                               full_emb_capacity=emb.shape[0],
                               backmode=args['backwardMode'],
                               distribution=args['distribution'],
                               alpha=args['zipf_alpha'],
                               )
    print("Construct PerfSampler done", flush=True)

    perf_sampler.Prefill()
    timer_geninput = Timer("GenInput")
    timer_Forward = Timer("Forward")
    timer_ForwardNN = Timer("forward: NN")
    timer_Backward = Timer("Backward")
    timer_Optimize_HBM = Timer("Optimize:HBM")
    timer_Optimize_DRAM = Timer("Optimize:DRAM")
    timer_onestep = Timer(f"OneStep")

    print("Before Training", flush=True)

    start_barrier = recstore.MultiProcessBarrierFactory.Create(
        "start_barrier", dist.get_world_size())
    start_barrier.Wait()

    exp_all_start_time = time.time()
    for _ in for_range:
        if rank == 0 and _ % 10 == 0:
            exp_all_now = time.time()
            if exp_all_now - exp_all_start_time > 90:
                break

        timer_onestep.start()
        sparse_opt.zero_grad()
        dist_opt.zero_grad()

        # print(f"========== Step {_} ========== ", flush=True)

        if with_perf and _ == warmup_iters:
            if rank == 0:
                if with_pyinstrucment:
                    pyinstruct_profiler.start()

                if with_cudaPerf:
                    print("cudaProfilerStart")
                    th.cuda.cudart().cudaProfilerStart()

                if with_torchPerf:
                    torch_profiler.start()

        if with_perf and _ == warmup_iters + 2:
            if rank == 0:
                if with_pyinstrucment:
                    pyinstruct_profiler.stop()
                    pyinstruct_profiler.print()
                if with_cudaPerf:
                    print("cudaProfilerStop")
                    th.cuda.cudart().cudaProfilerStop()
                if with_torchPerf:
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace("trace.json")
            break

        timer_geninput.start()
        input_keys = next(perf_sampler)
        start_barrier.Wait()
        timer_geninput.stop()

        timer_Forward.start()
        with xmh_nvtx_range(f"Step{_}:forward", condition=rank == 0 and _ >= warmup_iters and args['with_perf']):
            embed_value = abs_emb.forward(input_keys)
            timer_ForwardNN.start()
            loss = embed_value.sum(-1).sum(-1)
            timer_ForwardNN.stop()
        timer_Forward.stop()

        timer_Backward.start()
        loss.backward()
        timer_Backward.stop()

        if DIFF_TEST:
            # diff test begin
            std_embed_value = std_emb.forward(input_keys)
            std_loss = std_embed_value.sum(-1).sum(-1)
            std_loss.backward()
            # XLOG.cdebug(
            #     f"{rank}:std_embed_value {std_embed_value}")
            # XLOG.cdebug(
            #     f"{rank}:embed_value {embed_value}")
            # XLOG.cdebug(
            #     f"{rank}:full_emb {abs_emb.full_emb.get_shm_tensor()}")
            assert (torch.allclose(
                embed_value.cpu(), std_embed_value.cpu())), "forward is error"
            assert torch.allclose(loss, std_loss)
            # diff done

        timer_Optimize_HBM.start()
        sparse_opt.step()
        timer_Optimize_HBM.stop()

        timer_Optimize_DRAM.start()
        dist_opt.step()
        timer_Optimize_DRAM.stop()

        if (_ % args['log_interval']) == (args['log_interval']-1):
            end = time.time()
            print(
                f"Step{_}:rank{rank}, time: {end-start:.3f}, per_step: {(end-start)/(_-start_step+1):.6f}", flush=True)
            start = time.time()
            start_step = _
            if rank == 0:
                Timer.Report()

        timer_onestep.stop()

    print("Successfully xmh", flush=True)


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(5678)
    # print("wait debugpy connect", flush=True)
    # debugpy.wait_for_client()

    args = get_run_config()
    main_routine(args, routine_local_cache_helper)
