from __future__ import annotations

import copy
import unittest

import torch

from model_zoo.rs_demo.models.dlrm import build_hybrid_dense_arch
from model_zoo.rs_demo.models.utils import (
    flatten_embedded_sparse_grad_for_recstore,
    prepare_hybrid_dlrm_input,
    reshape_recstore_embeddings_for_dlrm,
    run_hybrid_backward,
)
from model_zoo.rs_demo.runtime.dense_graph import CapturedDlrmDense
from model_zoo.rs_demo.runtime.gradient_buffer import PlannedDenseGradientBuffer


class TestHybridDlrm(unittest.TestCase):
    def test_reshape_recstore_embeddings_for_dlrm_restores_bfd_layout(self) -> None:
        embeddings = torch.tensor(
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
                [40.0, 41.0],
                [50.0, 51.0],
                [60.0, 61.0],
            ]
        )

        embedded_sparse = reshape_recstore_embeddings_for_dlrm(
            embeddings=embeddings,
            batch_rows=2,
            num_sparse_features=3,
        )

        self.assertEqual(tuple(embedded_sparse.shape), (2, 3, 2))
        self.assertTrue(
            torch.equal(
                embedded_sparse,
                torch.tensor(
                    [
                        [[10.0, 11.0], [30.0, 31.0], [50.0, 51.0]],
                        [[20.0, 21.0], [40.0, 41.0], [60.0, 61.0]],
                    ]
                ),
            )
        )

    def test_flatten_embedded_sparse_grad_for_recstore_restores_feature_major_order(self) -> None:
        embedded_sparse_grad = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ]
        )

        flat = flatten_embedded_sparse_grad_for_recstore(embedded_sparse_grad)

        self.assertEqual(tuple(flat.shape), (6, 2))
        self.assertTrue(
            torch.equal(
                flat,
                torch.tensor(
                    [
                        [1.0, 2.0],
                        [7.0, 8.0],
                        [3.0, 4.0],
                        [9.0, 10.0],
                        [5.0, 6.0],
                        [11.0, 12.0],
                    ]
                ),
            )
        )

    def test_build_hybrid_dense_arch_matches_dlrm_output_shape(self) -> None:
        model = build_hybrid_dense_arch(
            torch=torch,
            dense_in_features=13,
            embedding_dim=4,
            num_sparse_features=3,
            dense_arch_layer_sizes=[8, 4],
            over_arch_layer_sizes=[16, 1],
            device=torch.device("cpu"),
        )
        dense = torch.randn(2, 13)
        embedded_sparse = torch.randn(2, 3, 4, requires_grad=True)

        logits = model(dense, embedded_sparse)

        self.assertEqual(tuple(logits.shape), (2, 1))
        loss = logits.sum()
        dense_grads = torch.autograd.grad(loss, list(model.parameters()) + [embedded_sparse])
        self.assertEqual(len(dense_grads), len(list(model.parameters())) + 1)
        self.assertIsInstance(model, torch.nn.Module)

    def test_run_hybrid_backward_populates_parameter_grads_via_backward(self) -> None:
        model = torch.nn.Linear(4, 1)
        embedded_sparse = torch.randn(2, 4, requires_grad=True)
        loss = model(embedded_sparse).sum()

        embedded_sparse_grad = run_hybrid_backward(
            loss=loss,
            embedded_sparse=embedded_sparse,
            dense_module=model,
            torch=torch,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(embedded_sparse_grad.shape), tuple(embedded_sparse.shape))
        for param in model.parameters():
            self.assertIsNotNone(param.grad)

    def test_split_backward_can_propagate_grad_to_non_leaf_sparse_source(self) -> None:
        model = torch.nn.Linear(4, 1)
        leaf_sparse = torch.randn(2, 4, requires_grad=True)
        embedded_sparse_source = leaf_sparse * 2.0
        dense_batch = torch.randn(2, 13)
        labels_batch = torch.ones(2, 1)

        _, embedded_sparse, _ = prepare_hybrid_dlrm_input(
            dense_batch=dense_batch,
            embedded_sparse_source=embedded_sparse_source,
            labels_batch=labels_batch,
            torch=torch,
            device=torch.device("cpu"),
            detach_sparse=True,
        )
        loss = model(embedded_sparse).sum()

        embedded_sparse_grad = run_hybrid_backward(
            loss=loss,
            embedded_sparse=embedded_sparse,
            dense_module=model,
            torch=torch,
            device=torch.device("cpu"),
        )
        embedded_sparse_source.backward(embedded_sparse_grad)

        self.assertEqual(tuple(embedded_sparse_grad.shape), tuple(embedded_sparse.shape))
        self.assertIsNotNone(leaf_sparse.grad)

    def test_dense_gradient_buffer_matches_eager_sgd_across_steps(self) -> None:
        reference = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
        candidate = copy.deepcopy(reference)
        buffer = PlannedDenseGradientBuffer(candidate)
        self.addCleanup(buffer.close)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=0.1)
        for step in range(3):
            inputs = torch.arange(12, dtype=torch.float32).view(4, 3) / (step + 1)
            reference_optimizer.zero_grad(set_to_none=True)
            buffer.begin_step()
            reference(inputs).square().mean().backward()
            candidate(inputs).square().mean().backward()
            buffer.reduce()
            for param, view in zip(candidate.parameters(), buffer.views):
                self.assertIs(param.grad, view)
            reference_optimizer.step()
            candidate_optimizer.step()
            for expected, actual in zip(reference.parameters(), candidate.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            candidate_optimizer.zero_grad(set_to_none=True)

    def test_dense_gradient_buffer_rejects_unused_parameters_and_replaced_views(self) -> None:
        module = torch.nn.Linear(3, 2)
        buffer = PlannedDenseGradientBuffer(module)
        self.addCleanup(buffer.close)
        buffer.begin_step()
        module.weight.sum().backward()
        with self.assertRaisesRegex(RuntimeError, "participate"):
            buffer.reduce()
        buffer.begin_step()
        module(torch.ones(1, 3)).sum().backward()
        module.weight.grad = module.weight.grad.clone()
        with self.assertRaisesRegex(RuntimeError, "gradient view"):
            buffer.reduce()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graphs require a CUDA device")
    def test_captured_dlrm_matches_eager_sgd_across_batch_shapes(self) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        reference = build_hybrid_dense_arch(torch, 13, 4, 3, [8, 4], [16, 1], device)
        candidate = copy.deepcopy(reference)
        buffer = PlannedDenseGradientBuffer(candidate)
        self.addCleanup(buffer.close)
        initial_state = {name: value.clone() for name, value in candidate.state_dict().items()}
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        captured = CapturedDlrmDense(
            candidate, criterion, buffer, batch_sizes=(8, 3),
            num_features=3, embedding_dim=4,
        )
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))
        for name, value in candidate.state_dict().items():
            torch.testing.assert_close(value, initial_state[name], rtol=0, atol=0)
        self.assertIs(captured.flat, buffer.flat)
        self.assertIs(captured.views, buffer.views)
        gradient_pointers = [view.data_ptr() for view in buffer.views]
        generator = torch.Generator(device=device).manual_seed(20261025)
        reference_sparse = torch.randn(8, 3, 4, device=device, generator=generator) * 0.2
        candidate_sparse = reference_sparse.clone()
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=0.01)

        # Both shapes update prefixes of the same sparse state and live dense weights.
        for batch_size in (8, 3, 8):
            dense = torch.randn(batch_size, 13, device=device, generator=generator)
            labels = torch.randint(2, (batch_size, 1), device=device, generator=generator).float()
            sparse = reference_sparse[:batch_size].detach().requires_grad_(True)
            reference_optimizer.zero_grad(set_to_none=True)
            candidate_optimizer.zero_grad(set_to_none=True)
            reference_loss = criterion(reference(dense, sparse), labels)
            reference_loss.backward()
            loss, sparse_grad = captured(dense, candidate_sparse[:batch_size], labels)
            buffer.reduce()
            torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
            torch.testing.assert_close(sparse_grad, sparse.grad, rtol=0, atol=0)
            for expected, actual, view, pointer in zip(
                reference.parameters(), candidate.parameters(), buffer.views, gradient_pointers,
            ):
                self.assertIs(actual.grad, view)
                self.assertEqual(actual.grad.data_ptr(), pointer)
                torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
            reference_optimizer.step()
            candidate_optimizer.step()
            with torch.no_grad():
                reference_sparse[:batch_size].add_(sparse.grad, alpha=-0.01)
                candidate_sparse[:batch_size].add_(sparse_grad, alpha=-0.01)
            for expected, actual in zip(reference.parameters(), candidate.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(candidate_sparse, reference_sparse, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graphs require a CUDA device")
    def test_planned_graph_matches_dynamic_lookup_and_duplicate_gradient_packing(self) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        reference = build_hybrid_dense_arch(torch, 13, 4, 3, [8, 4], [16, 1], device)
        candidate = copy.deepcopy(reference)
        buffer = PlannedDenseGradientBuffer(candidate)
        self.addCleanup(buffer.close)
        generator = torch.Generator(device=device).manual_seed(20261026)
        reference_values = torch.randn(11, 4, device=device, generator=generator) * 0.2
        resident_values = reference_values.clone()
        criterion = torch.nn.BCEWithLogitsLoss()
        captured = CapturedDlrmDense(
            candidate, criterion, buffer, batch_sizes=(8, 3), num_features=3, embedding_dim=4,
            resident_values=resident_values, payload_capacities={8: 5, 3: 7},
        )
        torch.testing.assert_close(resident_values, reference_values, rtol=0, atol=0)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=0.01)
        for step, (batch_size, unique_count, stride) in enumerate(((8, 5, 1), (3, 7, 2), (8, 3, 1))):
            # Duplicate IDs, changing maps, and a smaller payload on the same graph.
            lookup_rows = (torch.arange(batch_size * 3, device=device) * stride + step) % unique_count
            lookup_rows = (lookup_rows * 2 + 1) % reference_values.shape[0]
            active_rows, pooled_rows = lookup_rows.unique(return_inverse=True)
            dense = torch.randn(batch_size, 13, device=device, generator=generator)
            labels = torch.randint(2, (batch_size, 1), device=device, generator=generator).float()
            sparse = reference_values.index_select(0, lookup_rows).view(batch_size, 3, 4).requires_grad_(True)
            reference_optimizer.zero_grad(set_to_none=True)
            candidate_optimizer.zero_grad(set_to_none=True)
            reference_loss = criterion(reference(dense, sparse), labels)
            reference_loss.backward()
            reference_payload = torch.zeros(active_rows.numel(), 4, device=device)
            reference_payload.index_add_(0, pooled_rows, sparse.grad.reshape(-1, 4))
            loss, payload = captured(
                dense, None, labels, lookup_rows=lookup_rows, pooled_rows=pooled_rows,
                payload_rows=active_rows.numel(),
            )
            buffer.reduce()
            torch.testing.assert_close(loss, reference_loss, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(payload, reference_payload, rtol=1e-5, atol=1e-6)
            self.assertEqual(payload.shape, reference_payload.shape)
            for expected, actual, view in zip(reference.parameters(), candidate.parameters(), buffer.views):
                self.assertIs(actual.grad, view)
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
            reference_optimizer.step()
            candidate_optimizer.step()
            with torch.no_grad():
                reference_values.index_add_(0, active_rows, reference_payload, alpha=-0.01)
                resident_values.index_add_(0, active_rows, payload, alpha=-0.01)
            for expected, actual in zip(reference.parameters(), candidate.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(resident_values, reference_values, rtol=1e-5, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "payload count"):
            captured(dense, None, labels, lookup_rows=lookup_rows, pooled_rows=pooled_rows, payload_rows=6)
        with self.assertRaisesRegex(ValueError, "batch size"):
            captured(dense[:2], None, labels[:2], lookup_rows=lookup_rows, pooled_rows=pooled_rows, payload_rows=3)


if __name__ == "__main__":
    unittest.main()
