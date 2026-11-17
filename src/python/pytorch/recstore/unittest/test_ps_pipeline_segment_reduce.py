"""Stable segment plans preserve the public ticket and sparse SGD ordering."""
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from ..ps_pipeline import PSPipeline, PSPipelineSGD
from .test_ps_pipeline import _PS, _pipeline
from .test_ps_pipeline_dependency_split import _split, _prepare, _consume


class TestSegmentPlanCPU(unittest.TestCase):
    def test_ticket_matches_unique_with_overlapping_namespaces_and_tail(self):
        generator = torch.Generator().manual_seed(20260330)
        pipeline = PSPipeline(None, _PS(), table_offsets={"a": 0, "b": 1, "c": 0},
                              master_table_name="table", embedding_dim=2, device="cpu")
        try:
            for batch_size in (0, 1, 5, 129):
                batch = torch.randint(0, 8, (batch_size, 3), generator=generator)
                old, _, absent = pipeline._build_cpu_ticket(0, None, batch)
                self.assertIsNone(absent)
                pipeline._segment_reduce = True  # Exercise CPU planning without a CUDA context.
                ticket, _, plan = pipeline._build_cpu_ticket(0, None, batch)
                pipeline._segment_reduce = False
                self.assertEqual(len(ticket), 3)
                self.assertEqual(ticket[2], old[2])
                self.assertTrue(torch.equal(ticket[0], old[0]))
                self.assertTrue(torch.equal(ticket[1], old[1]))
                raw = (batch + pipeline._offsets).reshape(-1)
                self.assertEqual(sorted(plan.permutation.tolist()), list(range(raw.numel())))
                self.assertEqual(int(plan.offsets[0]), 0)
                self.assertEqual(int(plan.offsets[-1]), raw.numel())
                for key, begin, end in zip(ticket[0], plan.offsets[:-1], plan.offsets[1:]):
                    positions = plan.permutation[begin:end]
                    self.assertEqual(positions.tolist(), torch.where(raw == key)[0].tolist())
            self.assertEqual(pipeline.prepared_steps, 0)
            self.assertFalse(pipeline._segment_plans)
        finally:
            pipeline.shutdown(flush=False)

    def test_cpu_and_disabled_modes_do_not_prepare_plans(self):
        for setting in ("0", "1"):
            with self.subTest(setting=setting), mock.patch.dict(os.environ, RECSTORE_PS_SEGMENT_REDUCE=setting):
                pipeline = _pipeline(_PS())
                try:
                    self.assertFalse(pipeline._segment_reduce)
                    ticket = pipeline.on_batch_prepared(0, None, torch.tensor([[1], [1]]), {})
                    pipeline.before_lookup(0, None, ticket, {})
                    optimizer = PSPipelineSGD(SimpleNamespace(reset_trace=lambda: None), pipeline)
                    with mock.patch.object(torch, "segment_reduce", side_effect=AssertionError("CPU fallback")):
                        optimizer.record_pooled_grad(torch.ones((2, 1, 2)))
                    self.assertEqual(optimizer._grads.tolist(), [[2., 2.]])
                    self.assertFalse(pipeline._segment_plans)
                finally:
                    pipeline.shutdown(flush=False)

    def test_two_prepared_plans_are_released_after_submission_and_abort(self):
        with _split(_PS(), total_steps=3) as pipeline:
            pipeline._segment_reduce = True
            first = _prepare(pipeline, 0, [1, 1, 2])
            second = _prepare(pipeline, 1, [2, 4])
            self.assertEqual(set(pipeline._segment_plans), {0, 1})
            _consume(pipeline, 0, first)
            self.assertEqual(set(pipeline._segment_plans), {1})
            self.assertEqual(pipeline._current_segment_plan.step, 0)
            pipeline.submit_sparse_grad(first[1][0], torch.ones((2, 2)))
            self.assertIsNone(pipeline._current_segment_plan)
            third = _prepare(pipeline, 2, [])
            self.assertEqual(set(pipeline._segment_plans), {1, 2})
            _consume(pipeline, 1, second)
            pipeline.submit_sparse_grad(second[1][0], torch.ones((2, 2)))
            _consume(pipeline, 2, third)
            self.assertEqual(pipeline._current_segment_plan.offsets.tolist(), [0])
            pipeline.shutdown(flush=False)
            self.assertFalse(pipeline._segment_plans)
            self.assertIsNone(pipeline._current_segment_plan)

    def test_stale_ticket_is_rejected_and_failed_submission_keeps_plan_until_abort(self):
        pipeline = _pipeline(_PS())
        pipeline._segment_reduce = True
        try:
            ticket = pipeline.on_batch_prepared(0, None, torch.tensor([[1], [1]]), {})
            with self.assertRaisesRegex(RuntimeError, "segment plan"):
                pipeline.before_lookup(0, None, (ticket[0].clone(), *ticket[1:]), {})
            self.assertIn(0, pipeline._segment_plans)
            pipeline.before_lookup(0, None, ticket, {})
            with mock.patch.object(pipeline.kv_client, "sparse_gradient_commit_async",
                                   side_effect=RuntimeError("submission failed")):
                with self.assertRaisesRegex(RuntimeError, "submission failed"):
                    pipeline.submit_sparse_grad(ticket[0], torch.ones((1, 2)))
            self.assertIsNotNone(pipeline._current_segment_plan)
        finally:
            pipeline.shutdown(flush=False)
        self.assertFalse(pipeline._segment_plans)
        self.assertIsNone(pipeline._current_segment_plan)

    def test_read_scheduling_and_staging_failures_release_plans_on_shutdown(self):
        for failure in ("_queue_prepared_read", "_stage_rows"):
            with self.subTest(failure=failure):
                pipeline = _pipeline(_PS())
                pipeline._segment_reduce = True
                try:
                    with mock.patch.object(pipeline, failure, side_effect=RuntimeError("injected failure")):
                        with self.assertRaisesRegex(RuntimeError, "injected failure"):
                            ticket = pipeline.on_batch_prepared(0, None, torch.tensor([[1]]), {})
                            pipeline.before_lookup(0, None, ticket, {})
                    if failure == "_queue_prepared_read":
                        self.assertEqual(pipeline.prepared_steps, 0)
                        self.assertFalse(pipeline._segment_plans)
                    else:
                        self.assertIsNotNone(pipeline._current_segment_plan)
                finally:
                    pipeline.shutdown(flush=False)
                self.assertFalse(pipeline._segment_plans)
                self.assertIsNone(pipeline._current_segment_plan)


@unittest.skipUnless(os.environ.get("RECSTORE_TEST_CUDA") == "1" and torch.cuda.is_available(),
                     "set RECSTORE_TEST_CUDA=1 in an available GPU window")
class TestSegmentPlanCUDA(unittest.TestCase):
    def test_disabled_or_unavailable_operation_keeps_cuda_fallback(self):
        for setting, operation in (("0", torch.segment_reduce), ("1", None)):
            with self.subTest(setting=setting), mock.patch.dict(
                    os.environ, RECSTORE_PS_SEGMENT_REDUCE=setting), mock.patch.object(torch, "segment_reduce", operation):
                pipeline = PSPipeline(None, _PS(), table_offsets={"a": 0},
                                      master_table_name="table", embedding_dim=2, device="cuda")
                try:
                    self.assertFalse(pipeline._segment_reduce)
                    metrics = {}
                    ticket = pipeline.on_batch_prepared(0, None, torch.tensor([[1], [1], [2]]), metrics)
                    self.assertFalse(pipeline._segment_plans)
                    pipeline.before_lookup(0, None, ticket, metrics)
                    optimizer = PSPipelineSGD(SimpleNamespace(reset_trace=lambda: None), pipeline)
                    optimizer.record_pooled_grad(torch.ones((3, 1, 2), device="cuda"))
                    self.assertEqual(optimizer._grads.cpu().tolist(), [[2., 2.], [1., 1.]])
                    self.assertEqual(metrics["ps_pipeline_segment_reduce"], 0)
                finally:
                    pipeline.shutdown(flush=False)

    def test_bitwise_gradient_order_modes_tail_empty_and_fallback(self):
        original_mode = torch.get_deterministic_debug_mode()
        generator = torch.Generator().manual_seed(20260330)
        try:
            for dimension in (1, 2, 7, 32, 33, 128):
                for mode in (0, 1, 2):
                    with self.subTest(dimension=dimension, mode=mode), mock.patch.dict(
                            os.environ, RECSTORE_PS_SEGMENT_REDUCE="1"):
                        torch.set_deterministic_debug_mode(mode)
                        ps = _PS()
                        ps.rows = torch.ones((16, dimension))
                        native_submit = ps.sparse_gradient_commit_async
                        ps.sparse_gradient_commit_async = lambda name, ids, grads, **kwargs: native_submit(
                            name, ids, grads.cpu(), **kwargs)
                        pipeline = PSPipeline(None, ps, table_offsets={"a": 0, "b": 1, "c": 0},
                                              master_table_name="table", embedding_dim=dimension, device="cuda")
                        optimizer = PSPipelineSGD(SimpleNamespace(reset_trace=lambda: None), pipeline)
                        try:
                            for step, batch_size in enumerate((257, 5, 1, 0)):
                                batch = torch.randint(0, 8, (batch_size, 3), generator=generator)
                                batch[:, 0] = 0
                                metrics = {}
                                ticket = pipeline.on_batch_prepared(step, None, batch, metrics)
                                # Missing plans use the old reduction even when the switch is enabled.
                                if step == 2:
                                    pipeline._segment_plans.clear()
                                pipeline.before_lookup(step, None, ticket, metrics)
                                grad = torch.randn((batch_size, 3, dimension), generator=generator)
                                if batch_size >= 5:
                                    grad[:5, 0, :] = torch.tensor([1e20, 3.25, -1e20, 1., -0.])[:, None]
                                    if dimension >= 2:
                                        grad[:5, 0, 1] = torch.tensor(
                                            [1, -2147483647, 0, -2147483648, 1], dtype=torch.int32).view(torch.float32)
                                grad = grad.cuda()
                                if step == 0:
                                    grad = grad.transpose(1, 2).contiguous().transpose(1, 2)
                                expected = torch.zeros((ticket[0].numel(), dimension), device="cuda")
                                torch.set_deterministic_debug_mode(2)
                                expected.index_add_(0, pipeline._current_inverse, grad.reshape(-1, dimension))
                                torch.set_deterministic_debug_mode(mode)
                                for invalid in (grad.double(), grad.cpu()):
                                    with self.assertRaises(ValueError):
                                        optimizer.record_pooled_grad(invalid)
                                optimizer.record_pooled_grad(grad)
                                self.assertTrue(torch.equal(expected.view(torch.int32), optimizer._grads.view(torch.int32)))
                                self.assertEqual(torch.get_deterministic_debug_mode(), mode)
                                self.assertEqual(metrics["ps_pipeline_segment_reduce"], int(dimension >= 2 and step != 2))
                                optimizer.step()
                                self.assertIsNone(pipeline._current_segment_plan)
                            pipeline.finish_training()
                        finally:
                            pipeline.shutdown(flush=False)
        finally:
            torch.set_deterministic_debug_mode(original_mode)

    def test_owner_pipeline_does_not_enable_segment_reduce(self):
        class OwnerPipeline(PSPipeline):
            owner_merged_grad = True
        with mock.patch.dict(os.environ, RECSTORE_PS_SEGMENT_REDUCE="1"):
            pipeline = OwnerPipeline(None, _PS(), table_offsets={"a": 0},
                                     master_table_name="table", embedding_dim=2, device="cuda")
            try:
                self.assertFalse(pipeline._segment_reduce)
            finally:
                pipeline.shutdown(flush=False)

    def test_failed_reduction_preserves_mode_and_plan_until_abort(self):
        original_mode = torch.get_deterministic_debug_mode()
        try:
            for mode in (0, 1, 2):
                with mock.patch.dict(os.environ, RECSTORE_PS_SEGMENT_REDUCE="1"):
                    pipeline = PSPipeline(None, _PS(), table_offsets={"a": 0},
                                          master_table_name="table", embedding_dim=2, device="cuda")
                try:
                    ticket = pipeline.on_batch_prepared(0, None, torch.tensor([[1], [1]]), {})
                    pipeline.before_lookup(0, None, ticket, {})
                    optimizer = PSPipelineSGD(SimpleNamespace(reset_trace=lambda: None), pipeline)
                    torch.set_deterministic_debug_mode(mode)
                    with mock.patch.object(torch, "segment_reduce", side_effect=RuntimeError("reduction failed")):
                        with self.assertRaisesRegex(RuntimeError, "reduction failed"):
                            optimizer.record_pooled_grad(torch.ones((2, 1, 2), device="cuda"))
                    self.assertEqual(torch.get_deterministic_debug_mode(), mode)
                    self.assertIsNotNone(pipeline._current_segment_plan)
                finally:
                    pipeline.shutdown(flush=False)
                self.assertIsNone(pipeline._current_segment_plan)
        finally:
            torch.set_deterministic_debug_mode(original_mode)
