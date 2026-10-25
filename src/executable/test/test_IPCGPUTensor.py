import torch
import torch as th

from controller_process import KGCacheControllerWrapperBase

import recstore

import torch.multiprocessing as mp
import tqdm


def worker_main(worker_id, barrier, d):
    th.cuda.set_device(0)

    if worker_id == 0:
        # master process: change the tensor from 1 to 100
        gpu_tensor = recstore.IPCTensorFactory.NewIPCGPUTensor(
            "gpu0_tensor", (1000,), th.int64, 0)
        gpu_tensor.zero_()
        gpu_tensor[:10] = 1

        barrier.wait()

        cpu_tensor = recstore.IPCTensorFactory.NewIPCTensor(
            "cpu0_tensor", (1000,), th.int64)
        cpu_tensor.zero_()
        cpu_tensor[:10] = 1
        barrier.wait()

    else:
        barrier.wait()
        gpu_tensor = recstore.IPCTensorFactory.FindIPCTensorFromName(
            "gpu0_tensor")
        print(f"in worker{worker_id} process", gpu_tensor[:10])
        gpu_tensor[:10] = 100
        print(f"in worker{worker_id} process", gpu_tensor[:10])

        barrier.wait()

        cpu_tensor = recstore.IPCTensorFactory.FindIPCTensorFromName(
            "cpu0_tensor")
        print(f"in worker{worker_id} process", cpu_tensor[:10])
        cpu_tensor[:10] = 100
        print(f"in worker{worker_id} process", cpu_tensor[:10])
        return


workers = []
nr_workers = 2


barrier = mp.Barrier(nr_workers)

KGCacheControllerWrapperBase.BeforeDDPInit()

with mp.Manager() as manager:
    d = manager.dict()

    for worker_id in range(nr_workers):
        p = mp.Process(target=worker_main,
                       args=(worker_id, barrier, d))
        p.start()
        print(f"Worker {worker_id} pid={p.pid}")
        workers.append(p)

    for each in workers:
        each.join()

# print("in main process", shm_tensor[:10])
