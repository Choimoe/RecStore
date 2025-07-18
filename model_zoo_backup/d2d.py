import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import time

def main_worker(rank, world_size, args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:9001",
        world_size=world_size,
        rank=rank,
    )
    print("process begin", rank)

    for datatype in [torch.float,]:
        if rank == 0:
            print(f"Current datatype: {datatype}.")
            t = torch.rand(1024**3,dtype=datatype).to(torch.device('cuda',rank))      
            print("ready0")
            time.sleep(10)
            for _ in range(10000):
                with torch.cuda.nvtx.range("xmh send"):
                    dist.send(t,1)

        elif rank == 1:
            r = torch.rand(1024**3, dtype=datatype).to(torch.device('cuda',rank)) 
            print("ready1")
            time.sleep(10)
            for _ in range(10000):
                with torch.cuda.nvtx.range("xmh recv"):
                    dist.recv(r,0)

def main():
    mp.spawn(main_worker, nprocs=2, args=(2, 2))

if __name__ == "__main__":
    main()