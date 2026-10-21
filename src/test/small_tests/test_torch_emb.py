import torch
import torch.nn.functional as F
import time
import torch.optim as optim

import sys

from cache_common import KGExternelEmbedding

from torch.profiler import profile, record_function, ProfilerActivity

weight = torch.rand((int(1e6), 128), device="cpu").requires_grad_(True)



emb = KGExternelEmbedding(weight)

fake_tensor = torch.rand((1, 1),)

sparse_opt = optim.SGD(
    [fake_tensor], lr=0.01,)


emb.reg_opt(sparse_opt)

with_perf = False

# use_my_emb = False
use_my_emb = True



# def ProcessWorker(worker_id):
    


# for batch_size in [128, 512, 1024, 1536, 2048]:
for batch_size in [128*1000, 512*1000, 1024*1000, 1536*1000, 2048*1000]:
    mean = 0
    count = 0
    for _ in range(int(1e2)):
        sparse_opt.zero_grad()
        keys = torch.randint(0, int(1e6), (batch_size, ), device="cpu")
        keys = keys.cuda()
        
        if with_perf and _ == 10:
            torch_profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
            torch_profiler.start()

        
        start = time.time()

        if use_my_emb:
            value = emb.forward(keys)
        else:
            value = F.embedding(keys.cpu(), weight, sparse=True, padding_idx=None, scale_grad_by_freq=False,)
            # value = weight.index_select(dim = 0, index = keys)
            
        loss = value.sum(-1).sum(-1)
        loss.backward()

        if not use_my_emb:
            sparse_opt.step()

        end = time.time()
        mean = (mean * count + (end - start)) / (count + 1)
        
        if with_perf and _ == 10:
            torch_profiler.stop()
            torch_profiler.export_chrome_trace("trace.json")
            sys.exit(0)


    print(batch_size, mean*1e6, "us")
    print(batch_size, "TPS", 1/mean*batch_size/1e6, "M")
