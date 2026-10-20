"""Exercise the standalone import bootstrap and optimizer ordering without training."""

import ast
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import torch

from recstore.optimizer import SparseSGD


SCRIPT = Path(__file__).with_name("dlrm_main_single_day.py")


class SingleDayTopKTest(unittest.TestCase):
    def test_compression_import_works_after_standalone_path_bootstrap(self):
        # Execute the real bootstrap/import statements in their original order.
        tree = ast.parse(SCRIPT.read_text())
        names = {"REPO_ROOT", "RECSTORE_PATH", "DLRM_PATH"}
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in names
                for target in node.targets
            ):
                nodes.append(node)
            elif isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                if isinstance(node.test.left, ast.Name) and node.test.left.id in names:
                    nodes.append(node)
            elif isinstance(node, ast.ImportFrom) and node.module == (
                "python.pytorch.recstore.gradient_compression"
            ):
                nodes.append(node)
        code = "import os, sys\n__file__ = " + repr(str(SCRIPT.resolve())) + "\n"
        code += ast.unparse(ast.Module(body=nodes, type_ignores=[]))
        code += "\nassert callable(warmup_topk)\n"
        env = dict(os.environ, RECSTORE_DEFER_OPS_LOAD="1", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(
            [sys.executable, "-B", "-I", "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

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
