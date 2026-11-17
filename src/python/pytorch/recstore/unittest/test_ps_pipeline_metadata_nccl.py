"""CPU-only mocks for transport order and ownership, not NCCL correctness."""
from collections import deque
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
import os
import threading
import unittest
from unittest import mock

import torch

from ..ps_pipeline import PSPipeline
from ..ps_pipeline import pipeline as pipeline_module


class _InlineExecutor:
    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except BaseException as error:
            future.set_exception(error)
        return future


@contextmanager
def _transport(peer_ids=(), *, executor=None, header_change=None, record_error=False):
    p = object.__new__(PSPipeline)
    p.device = torch.device("cuda:0")
    p._metadata_transfers = deque()
    p._metadata_capacity = 9
    p._metadata_submit_thread = threading.get_ident()
    p._metadata_group = object()
    p._metadata_worker = executor or _InlineExecutor()
    p._world, p._total_steps, p._session = 2, 12, 91
    p._submission_error = None
    log = []

    class Event:
        def record(self, stream):
            log.append("record")
            if record_error:
                raise RuntimeError("record failed")

        def synchronize(self):
            log.append("event_wait")

    def gather(outputs, source, **kwargs):
        assert threading.get_ident() == p._metadata_submit_thread
        assert kwargs == {"group": p._metadata_group, "async_op": True}
        assert len(p._metadata_transfers) <= 2
        log.append("gather")
        outputs[0].copy_(source)
        outputs[1].zero_()
        outputs[1][:4].copy_(source[:4])
        outputs[1][1] = len(peer_ids)
        outputs[1][4:4 + len(peer_ids)] = torch.tensor(peer_ids, dtype=torch.int64)
        if header_change:
            outputs[1][header_change[0]] = header_change[1]
        return mock.Mock(wait=lambda: log.append("work_wait"))

    def cpu_alloc(fn):
        def alloc(*args, **kwargs):
            kwargs.pop("device", None)
            kwargs.pop("pin_memory", None)
            return fn(*args, **kwargs)
        return alloc

    with ExitStack() as stack:
        for name in ("empty", "zeros"):
            stack.enter_context(mock.patch.object(torch, name, cpu_alloc(getattr(torch, name))))
        stack.enter_context(mock.patch.object(pipeline_module.dist, "all_gather", gather))
        stack.enter_context(mock.patch.object(torch.cuda, "Event", Event))
        stack.enter_context(mock.patch.object(torch.cuda, "current_stream", lambda device: "main"))
        stack.enter_context(mock.patch.object(torch.cuda, "synchronize", lambda device: log.append("device_wait")))
        yield p, log


class TestMetadataNCCLCPU(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {
            "RECSTORE_PS_METADATA_NCCL": "0", "RECSTORE_PS_SEGMENT_REDUCE": "0",
            "RECSTORE_PS_GPU_TICKET": "0", "RECSTORE_PS_DEVICE_MERGE": "0",
            "RECSTORE_DENSE_REDUCE_OVERLAP": "0", "RECSTORE_PS_PREPARE_DURING_DENSE": "0",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_default_off_keeps_gloo_and_needs_no_batch_bound(self):
        group = object()
        with mock.patch.dict(os.environ, {"RECSTORE_PS_METADATA_NCCL": "0"}), \
             mock.patch.multiple(pipeline_module.dist, is_initialized=lambda: True,
                                 get_world_size=lambda: 2, get_rank=lambda: 0,
                                 broadcast=lambda *args, **kwargs: None,
                                 destroy_process_group=lambda value: None), \
             mock.patch.object(pipeline_module.dist, "new_group", return_value=group) as create:
            p = PSPipeline(None, None, table_offsets={"a": 0}, master_table_name="x",
                           embedding_dim=2, device="cpu", dependency_split=True, total_steps=1)
            try:
                self.assertFalse(p._metadata_nccl)
                self.assertEqual(create.call_args.kwargs["backend"], "gloo")
                self.assertEqual(p._metadata_capacity, 0)
            finally:
                p.shutdown(flush=False)

    def test_final_step_suppresses_unused_collective(self):
        with _transport() as (p, log):
            p.dependency_split = p._metadata_nccl = True
            p._total_steps = 1
            p._metadata = {}
            p._worker = _InlineExecutor()
            p._pull = lambda ids, previous: (ids, 0.0, 0.0)
            p._queue_prepared_read(0, torch.arange(2))
            self.assertEqual(log, [])
            self.assertFalse(p._metadata)
            self.assertFalse(p._metadata_transfers)
            self.assertTrue(p._read.done())

    def test_payload_union_order_and_counts(self):
        for local, remote in [([], []), ([], [0]), ([0, 2, 2**62], [0, 1, 2**63-1]),
                              (list(range(9)), list(range(9)))]:
            with self.subTest(local=local), _transport(remote) as (p, log):
                future = p._submit_nccl_dirty_ids(3, torch.tensor(local, dtype=torch.int64))
                ids, stats = future.result()
                self.assertEqual(ids.tolist(), sorted(set(local + remote)))
                self.assertEqual(ids.dtype, torch.int64)
                self.assertEqual(log, ["gather", "work_wait", "record", "event_wait"])
                record = p._metadata_transfers[0]
                self.assertEqual(record.host_send[:4].tolist(), [3, len(local), 12, 91])
                self.assertEqual(stats["ps_pipeline_metadata_sent_bytes"], 13 * 8)
                self.assertEqual(stats["ps_pipeline_local_dirty_rows"], len(local))
                p._reap_metadata_transfers()
                self.assertFalse(p._metadata_transfers)

    def test_header_validation(self):
        for change in ((0, 4), (1, -1), (1, 10), (2, 13), (3, 92)):
            with self.subTest(change=change), _transport(header_change=change) as (p, log):
                future = p._submit_nccl_dirty_ids(3, torch.tensor([0]))
                with self.assertRaises(RuntimeError):
                    future.result()
                with self.assertRaises(RuntimeError):
                    p._drain_metadata_transfers()
                self.assertFalse(p._metadata_transfers)

    def test_oversize_and_sticky_failure(self):
        with _transport() as (p, log):
            with self.assertRaisesRegex(ValueError, "exceed"):
                p._submit_nccl_dirty_ids(0, torch.arange(10))
            with self.assertRaisesRegex(ValueError, "exceed"):
                p._submit_nccl_dirty_ids(0, torch.arange(2))
            self.assertEqual(log, [])

    def test_cross_thread_submission_rejected_before_collective(self):
        with _transport() as (p, log):
            p._metadata_submit_thread = -1
            with self.assertRaisesRegex(RuntimeError, "training thread"):
                p._submit_nccl_dirty_ids(0, torch.arange(2))
            self.assertEqual(log, [])

    def test_executor_failure_keeps_buffers_until_event_drain(self):
        executor = mock.Mock(submit=mock.Mock(side_effect=RuntimeError("schedule failed")))
        with _transport(executor=executor) as (p, log):
            with self.assertRaisesRegex(RuntimeError, "schedule failed"):
                p._submit_nccl_dirty_ids(0, torch.arange(2))
            self.assertEqual(len(p._metadata_transfers), 1)
            self.assertIsNotNone(p._metadata_transfers[0].ready)
            p._drain_metadata_transfers()
            self.assertEqual(log[-1], "event_wait")
            self.assertFalse(p._metadata_transfers)

    def test_unpublished_event_requires_device_drain(self):
        with _transport(record_error=True) as (p, log):
            with self.assertRaisesRegex(RuntimeError, "record failed"):
                p._submit_nccl_dirty_ids(0, torch.arange(2))
            self.assertIsNone(p._metadata_transfers[0].ready)
            p._drain_metadata_transfers()
            self.assertEqual(log[-2:], ["work_wait", "device_wait"])

    def test_two_inflight_bound_waits_oldest(self):
        futures = []
        def defer(fn, transfer):
            future = Future()
            futures.append((future, fn, transfer))
            return future
        with _transport(executor=mock.Mock(submit=defer)) as (p, log):
            p._submit_nccl_dirty_ids(0, torch.arange(2))
            p._submit_nccl_dirty_ids(1, torch.arange(2))
            self.assertEqual(len(p._metadata_transfers), 2)
            first, fn, transfer = futures[0]
            timer = threading.Timer(0.01, lambda: first.set_result(fn(transfer)))
            timer.start()
            p._submit_nccl_dirty_ids(2, torch.arange(2))
            timer.join()
            self.assertEqual(len(p._metadata_transfers), 2)
            self.assertEqual([t.step for t in p._metadata_transfers], [1, 2])
            for future, fn, transfer in futures[1:]:
                future.set_result(fn(transfer))
            p._drain_metadata_transfers()

    def test_unsafe_dense_modes_and_cpu_are_rejected(self):
        options = dict(table_offsets={"a": 0}, master_table_name="x", embedding_dim=2,
                       dependency_split=True, total_steps=3, max_batch_size=9)
        with mock.patch.dict(os.environ, {"RECSTORE_PS_METADATA_NCCL": "1"}), \
             mock.patch.multiple(pipeline_module.dist, is_initialized=lambda: True,
                                 get_world_size=lambda: 2, get_rank=lambda: 0,
                                 get_backend=lambda: "nccl"):
            with self.assertRaisesRegex(ValueError, "base dependency-split"):
                PSPipeline(None, None, device="cpu", **options)
            for env in ("RECSTORE_DENSE_REDUCE_OVERLAP", "RECSTORE_PS_PREPARE_DURING_DENSE"):
                with mock.patch.dict(os.environ, {env: "1"}):
                    with self.assertRaisesRegex(ValueError, "asynchronous dense"):
                        PSPipeline(None, None, device="cuda:0", **options)
            for bound in (None, False, 0, -1):
                with self.subTest(bound=bound):
                    with self.assertRaisesRegex(ValueError, "positive explicit"):
                        PSPipeline(None, None, device="cuda:0", **{**options, "max_batch_size": bound})


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
