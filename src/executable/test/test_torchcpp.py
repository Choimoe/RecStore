from pathlib import Path

import torch as th


repo_root = Path(__file__).resolve().parents[3]
th.ops.load_library(str(repo_root / "build/lib/librecstore_pytorch_test.so"))


test1_op = th.ops.librecstore_pytorch_test.test1

full_emb = th.zeros((10, 3))
id = 1
grad = th.ones((1, 3))

test1_op(full_emb, id, grad)

print(full_emb)

th.cuda.synchronize()
