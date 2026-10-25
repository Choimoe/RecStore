"""Exercise optimizer ordering without training."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from recstore.optimizer import SparseSGD


SCRIPT = Path(__file__).with_name("dlrm_main_single_day.py")


class SingleDayTopKTest(unittest.TestCase):
    def test_rejected_topk_does_not_change_dense_weights_or_optimizer_state(self):
        tree = ast.parse(SCRIPT.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        batch_loop = next(
            node for node in ast.walk(main)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Tuple)
            and any(isinstance(item, ast.Name) and item.id == "batch_idx" for item in node.target.elts)
        )
        start = next(
            i for i, node in enumerate(batch_loop.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "raw_model" for target in node.targets)
        )
        end = next(
            i for i in range(start + 1, len(batch_loop.body))
            if isinstance(batch_loop.body[i], ast.If)
            and isinstance(batch_loop.body[i].test, ast.Name)
            and batch_loop.body[i].test.id == "use_cuda_timing"
        )
        # Use the actual training-loop block, including the precheck and both paths.
        code = compile(ast.Module(body=batch_loop.body[start + 1:end + 1], type_ignores=[]), str(SCRIPT), "exec")
        for use_cuda_timing in (False, True):
            with self.subTest(use_cuda_timing=use_cuda_timing):
                weight = torch.nn.Parameter(torch.tensor([1.0]))
                weight.grad = torch.ones_like(weight)
                dense = torch.optim.SGD([weight], lr=0.1, momentum=0.9)
                module = SimpleNamespace(
                    kv_client=object(), _config_names={}, _trace=[], _enable_fusion=False
                )
                sparse = SparseSGD([module], lr=0.1, keep_ratio=0.5)
                with self.assertRaisesRegex(NotImplementedError, "Embedding Fusion"):
                    exec(code, {
                        "sparse_optimizer": sparse,
                        "dense_optimizer": dense,
                        "use_cuda_timing": use_cuda_timing,
                    })
                torch.testing.assert_close(weight, torch.tensor([1.0]))
                self.assertEqual(dict(dense.state), {})
