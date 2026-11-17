"""GPU integer tickets retain CPU-visible plans and ordered preparation."""
from contextlib import contextmanager
import os
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from ..ps_pipeline import PSPipeline, PSPipelineSGD
from .test_ps_pipeline import _PS


_ENV = {'RECSTORE_PS_METADATA_NCCL':'0', 'RECSTORE_PS_DEVICE_MERGE':'0',
            'RECSTORE_DENSE_REDUCE_OVERLAP':'0', 'RECSTORE_PS_PREPARE_DURING_DENSE':'0',
            'RECSTORE_PS_SEGMENT_REDUCE':'1', 'RECSTORE_PS_GPU_TICKET':'1'}


def _pipeline(device='cpu', offsets=None, dimension=7, owner=False):
    cls = type('OwnerStub', (PSPipeline,), {'owner_merged_grad':True}) if owner else PSPipeline
    return cls(None, None, table_offsets=offsets or {'a':0,'b':1,'c':0},
               master_table_name='table', embedding_dim=dimension, device=device)


@contextmanager
def _cpu_cuda_mock():
    """Exercise integer tensor logic and dispatch; this does not emulate CUDA ordering."""
    tensor, empty, to, copy = torch.tensor, torch.empty, torch.Tensor.to, torch.Tensor.copy_
    state = {'copies':0, 'records':0, 'waits':0, 'drains':0}
    class Stream:
        def synchronize(self):
            state['drains'] += 1
    stream = Stream()
    class Event:
        def record(self, value):
            assert value is stream
            assert state['copies'] == 4, 'completion fence must follow all four D2H copies'
            state['records'] += 1
        def synchronize(self):
            assert state['records']
            state['waits'] += 1
    def cpu_tensor(*args, **kwargs):
        if torch.device(kwargs.get('device','cpu')).type == 'cuda':
            kwargs['device'] = 'cpu'
        return tensor(*args, **kwargs)
    def cpu_empty(*args, **kwargs):
        kwargs.pop('pin_memory', None)
        return empty(*args, **kwargs)
    def cpu_to(self, *args, **kwargs):
        args = list(args)
        if args and isinstance(args[0], (str, torch.device)) and torch.device(args[0]).type == 'cuda':
            args[0] = 'cpu'
        if 'device' in kwargs and torch.device(kwargs['device']).type == 'cuda':
            kwargs['device'] = 'cpu'
        return to(self, *args, **kwargs)
    def count_copy(self, *args, **kwargs):
        if kwargs.get('non_blocking'):
            state['copies'] += 1
        return copy(self, *args, **kwargs)
    with mock.patch.object(torch, 'tensor', cpu_tensor), mock.patch.object(torch, 'empty', cpu_empty), \
         mock.patch.object(torch.Tensor, 'to', cpu_to), mock.patch.object(torch.Tensor, 'pin_memory', lambda self:self.clone()), \
         mock.patch.object(torch.Tensor, 'copy_', count_copy), mock.patch.object(torch.cuda, 'current_stream', return_value=stream), \
         mock.patch.object(torch.cuda, 'Event', Event), mock.patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('no device barrier')):
        yield state


def _assert_same_ticket(test, expected, actual):
    old, _, old_plan = expected
    ticket, elapsed, plan = actual
    test.assertEqual(len(ticket), 3)
    test.assertEqual(ticket[2], old[2])
    test.assertGreaterEqual(elapsed, 0)
    for left, right in zip(old[:2], ticket[:2]):
        test.assertEqual(right.device.type, 'cpu')
        test.assertEqual(right.dtype, torch.int64)
        test.assertTrue(torch.equal(left, right))
    test.assertIs(plan.ids, ticket[0])
    test.assertEqual(plan.step, old_plan.step)
    for name in ('ids','permutation','offsets'):
        test.assertTrue(torch.equal(getattr(plan,name),getattr(old_plan,name)), name)


class TestGPUTicketCPU(unittest.TestCase):
    def test_gate_preserves_cpu_owner_disabled_and_unavailable_paths(self):
        cases = [('cpu',7,False,'1','1',torch.segment_reduce,False),
                 ('cuda',1,False,'1','1',torch.segment_reduce,False),
                 ('cuda',7,True,'1','1',torch.segment_reduce,False),
                 ('cuda',7,False,'0','1',torch.segment_reduce,False),
                 ('cuda',7,False,'1','0',torch.segment_reduce,False),
                 ('cuda',7,False,'1',None,torch.segment_reduce,False),
                 ('cuda',7,False,'1','1',None,False),
                 ('cuda',7,False,'1','1',torch.segment_reduce,True)]
        for device,dim,owner,segment,gpu,operation,enabled in cases:
            with self.subTest(device=device,dim=dim,owner=owner,segment=segment,gpu=gpu,operation=bool(operation)), \
                 mock.patch.dict(os.environ, {**_ENV,'RECSTORE_PS_SEGMENT_REDUCE':segment,'RECSTORE_PS_GPU_TICKET':gpu or '0'}), \
                 mock.patch.object(torch,'segment_reduce',operation), _cpu_cuda_mock():
                if gpu is None:
                    os.environ.pop('RECSTORE_PS_GPU_TICKET')
                p=_pipeline(device,dimension=dim,owner=owner)
                try:
                    self.assertEqual(p._gpu_ticket,enabled)
                    self.assertEqual(p._gpu_ticket_offsets is not None,enabled)
                finally:
                    p.shutdown(flush=False)
                self.assertIsNone(p._gpu_ticket_offsets)

    def test_helper_integer_results_and_host_fence(self):
        batches=[torch.empty((0,3),dtype=torch.int64), torch.zeros((1,3),dtype=torch.int64),
                 torch.zeros((65,3),dtype=torch.int64), torch.arange(7*6).reshape(7,6)[:,::2],
                 torch.arange(257*3).reshape(257,3)%7,
                 torch.arange(7*3,dtype=torch.int32).reshape(7,3)%7,
                 torch.arange(7*3,dtype=torch.float64).reshape(7,3)%7+0.75]
        for batch in batches:
            with self.subTest(shape=tuple(batch.shape),dtype=batch.dtype), mock.patch.dict(os.environ,_ENV), _cpu_cuda_mock() as state:
                p=_pipeline('cuda')
                try:
                    expected=p._build_cpu_ticket(9,None,batch)
                    _assert_same_ticket(self,expected,p._build_gpu_ticket(9,None,batch))
                    self.assertEqual((state['copies'],state['records'],state['waits'],state['drains']),(4,1,1,0))
                    self.assertEqual(p.prepared_steps,0)
                    self.assertFalse(p._segment_plans)
                finally:
                    p.shutdown(flush=False)

    def test_foreground_dispatch_adoption_and_cpu_builder_stays_cpu(self):
        with mock.patch.dict(os.environ,_ENV), _cpu_cuda_mock():
            p=_pipeline('cuda')
            try:
                batch=torch.tensor([[1,2,1],[1,0,2]])
                with mock.patch.object(p,'_build_gpu_ticket',side_effect=AssertionError('CPU builder must stay pure')):
                    p._build_cpu_ticket(0,None,batch)
                row={}
                with mock.patch.object(p,'_queue_prepared_read') as queue:
                    ticket=p.on_batch_prepared(0,None,batch,row)
                self.assertEqual(queue.call_count,1)
                self.assertIs(p._segment_plans[0].ids,ticket[0])
                self.assertEqual(p.prepared_steps,1)
                self.assertEqual(row['ps_pipeline_gpu_ticket'],1)
                self.assertGreaterEqual(row['lookup_ids_build_ms'],0)
            finally:
                p.shutdown(flush=False)
            self.assertFalse(p._segment_plans)

    def test_failure_does_not_adopt_or_retry_and_keeps_original_error(self):
        with mock.patch.dict(os.environ,_ENV), _cpu_cuda_mock() as state:
            p=_pipeline('cuda')
            original=RuntimeError('sort failure')
            try:
                with mock.patch.object(torch,'sort',side_effect=original), \
                     mock.patch.object(p,'_build_cpu_ticket',side_effect=AssertionError('no hidden CPU retry')), \
                     mock.patch.object(p,'_adopt_cpu_ticket') as adopt, \
                     mock.patch.object(p,'_queue_prepared_read') as queue:
                    with self.assertRaises(RuntimeError) as caught:
                        p.on_batch_prepared(0,None,torch.ones((4,3),dtype=torch.int64),{})
                self.assertIs(caught.exception,original)
                adopt.assert_not_called()
                queue.assert_not_called()
                self.assertEqual(p.prepared_steps,0)
                self.assertFalse(p._segment_plans)
                self.assertEqual(state['drains'],1)
            finally:
                p.shutdown(flush=False)

    def test_gpu_builder_rejects_worker_thread_before_cuda_operations(self):
        with mock.patch.dict(os.environ,_ENV), _cpu_cuda_mock():
            p=_pipeline('cuda')
            errors=[]
            def worker():
                try:
                    p._build_gpu_ticket(0,None,torch.empty((0,3),dtype=torch.int64))
                except BaseException as exc:
                    errors.append(exc)
            try:
                t=threading.Thread(target=worker)
                t.start()
                t.join(5)
                self.assertFalse(t.is_alive())
                self.assertEqual(len(errors),1)
                self.assertIn('training thread',str(errors[0]))
            finally:
                p.shutdown(flush=False)


@unittest.skipUnless(os.environ.get('RECSTORE_TEST_CUDA')=='1' and torch.cuda.is_available(),
                     'requires explicit GPU window and RECSTORE_TEST_CUDA=1')
class TestGPUTicketCUDA(unittest.TestCase):
    def test_foreground_cast_plan_consumption_and_sparse_update(self):
        original_mode = torch.get_deterministic_debug_mode()
        ps = _PS()
        ps.rows = torch.ones((16, 7))
        native_submit = ps.sparse_gradient_commit_async
        ps.sparse_gradient_commit_async = lambda name, ids, grads, **kwargs: native_submit(
            name, ids, grads.cpu(), **kwargs)
        with mock.patch.dict(os.environ, _ENV):
            p = PSPipeline(None, ps, table_offsets={"a": 0, "b": 1, "c": 0},
                           master_table_name="table", embedding_dim=7, device="cuda:0")
        optimizer = PSPipelineSGD(SimpleNamespace(reset_trace=lambda: None), p)
        try:
            for step, batch_size in enumerate((257, 5, 1, 0)):
                batch = (torch.arange(batch_size * 6, dtype=torch.float64)
                         .reshape(batch_size, 6)[:, ::2] % 7) + 0.75
                row = {}
                expected_ticket = p._build_cpu_ticket(step, None, batch)
                ticket = p.on_batch_prepared(step, None, batch, row)
                _assert_same_ticket(self, expected_ticket,
                                    (ticket, row["lookup_ids_build_ms"], p._segment_plans[step]))
                p.before_lookup(step, None, ticket, row)
                gradient = (torch.arange(batch_size * 3 * 7, dtype=torch.float32)
                            .reshape(batch_size, 3, 7) % 11).cuda()
                expected = torch.zeros((ticket[0].numel(), 7), device="cuda:0")
                torch.set_deterministic_debug_mode(2)
                expected.index_add_(0, p._current_inverse, gradient.reshape(-1, 7))
                torch.set_deterministic_debug_mode(original_mode)
                optimizer.record_pooled_grad(gradient)
                self.assertTrue(torch.equal(expected.view(torch.int32),
                                            optimizer._grads.view(torch.int32)))
                optimizer.step()
                self.assertIsNone(p._current_segment_plan)
                self.assertEqual(row["ps_pipeline_gpu_ticket"], 1)
            p.finish_training()
        finally:
            torch.set_deterministic_debug_mode(original_mode)
            p.shutdown(flush=False)

    def test_corner_cases_first_use_other_stream_and_cpu_outputs_ready(self):
        device=torch.device('cuda:0')
        construct=torch.cuda.Stream(device=device)
        consumer=torch.cuda.Stream(device=device)
        for offsets in ({'a':0,'b':1,'c':0},{'a':1<<55,'b':(1<<55)+4096}):
            with mock.patch.dict(os.environ,_ENV),torch.cuda.stream(construct):
                p=_pipeline(device,offsets)
            try:
                f=len(offsets)
                for step,n in enumerate((0,1,5,128,257,1024)):
                    batch=(torch.arange(n*f*2,dtype=torch.int32).reshape(n,f*2)[:,::2]%7)
                    if step % 2:
                        batch=batch.to(torch.float64)+0.75
                    expected=p._build_cpu_ticket(step,None,batch)
                    with torch.cuda.stream(consumer):
                        actual=p._build_gpu_ticket(step,None,batch)
                    # No extra CUDA synchronization may make outputs ready here.
                    _assert_same_ticket(self,expected,actual)
                    for host in (*actual[0][:2],actual[2].permutation,actual[2].offsets):
                        if host.numel():  # Empty tensors have no pinned allocation to query.
                            self.assertTrue(host.is_pinned())
                    snapshot=[x.clone() for x in (*actual[0][:2],actual[2].permutation,actual[2].offsets)]
                    zeros=torch.zeros_like(batch)
                    with torch.cuda.stream(consumer):
                        duplicate=p._build_gpu_ticket(step,None,zeros)
                    _assert_same_ticket(self,p._build_cpu_ticket(step,None,zeros),duplicate)
                    for old,current in zip(snapshot,(*actual[0][:2],actual[2].permutation,actual[2].offsets)):
                        self.assertTrue(torch.equal(old,current))
            finally:
                p.shutdown(flush=False)



if __name__ == '__main__':
    unittest.main()
