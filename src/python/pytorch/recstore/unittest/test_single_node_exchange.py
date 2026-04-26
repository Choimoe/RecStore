import unittest

import torch

from src.python.pytorch.recstore.single_node_exchange import (
    LookupEmbeddingResponsePayload,
    LookupIdsPayload,
    SparseGradPayload,
    exchange_lookup_embedding_responses,
    exchange_lookup_ids,
    exchange_sparse_grads,
    reassemble_lookup_embedding_responses,
)
import src.python.pytorch.recstore.single_node_exchange as exchange_module


class FakeDistBackend:
    def __init__(self, payloads_by_rank):
        self._payloads_by_rank = {
            int(rank): [payload.clone() for payload in payloads]
            for rank, payloads in payloads_by_rank.items()
        }
        self.calls = []

    def all_gather_object(self, output_objects, input_object):
        self.calls.append(type(input_object).__name__)
        gathered = self._payloads_by_rank.get(input_object.rank, [])
        if len(gathered) != len(output_objects):
            raise AssertionError(
                f"expected {len(output_objects)} gathered payloads, got {len(gathered)}"
            )
        for idx, payload in enumerate(gathered):
            output_objects[idx] = payload


class FakeTensorDistBackend(FakeDistBackend):
    def __init__(self, payloads_by_rank):
        super().__init__(payloads_by_rank)
        self._pending = {}

    def get_backend(self):
        return "gloo"

    def all_gather(self, output_tensors, input_tensor, group=None):
        call_idx = len(self.calls)
        self.calls.append(("tensor", tuple(input_tensor.shape), str(input_tensor.dtype), group))
        rank = None
        is_meta = input_tensor.dtype == torch.int64 and input_tensor.dim() == 1 and input_tensor.numel() == 3
        if is_meta:
            rank = int(input_tensor[0].item())
            self._pending[call_idx] = self._payloads_by_rank.get(rank, [])
        payloads = self._pending.get(call_idx - (0 if is_meta else 1), [])
        if not payloads:
            payloads = next(iter(self._pending.values()))
        if len(payloads) != len(output_tensors):
            raise AssertionError(
                f"expected {len(output_tensors)} gathered payloads, got {len(payloads)}"
            )
        for idx, payload in enumerate(payloads):
            if is_meta:
                rows = int(payload.source_ranks.numel()) if isinstance(payload, (LookupIdsPayload, SparseGradPayload)) else int(payload.requestor_ranks.numel())
                if isinstance(payload, LookupIdsPayload):
                    cols = 0
                elif isinstance(payload, LookupEmbeddingResponsePayload):
                    cols = int(payload.embeddings.shape[1])
                else:
                    cols = int(payload.grads.shape[1])
                output_tensors[idx].copy_(torch.tensor([payload.rank, rows, cols], dtype=torch.int64))
                continue
            if isinstance(payload, LookupIdsPayload):
                fields = (
                    payload.source_ranks,
                    payload.row_positions,
                    payload.fused_ids,
                )
            elif isinstance(payload, LookupEmbeddingResponsePayload):
                fields = (
                    payload.requestor_ranks,
                    payload.row_positions,
                    payload.embeddings,
                )
            else:
                fields = (
                    payload.source_ranks,
                    payload.row_positions,
                    payload.fused_ids,
                    payload.grads,
                )
            field = None
            for candidate in fields:
                if candidate.dtype == input_tensor.dtype and candidate.dim() == input_tensor.dim():
                    if candidate.dim() == 2 and candidate.shape[1] != input_tensor.shape[1]:
                        continue
                    field = candidate
                    break
            if field is None:
                raise AssertionError("unable to match gathered tensor field")
            output_tensors[idx].zero_()
            if field.numel() > 0:
                if field.dim() == 1:
                    output_tensors[idx][: field.numel()].copy_(field)
                else:
                    output_tensors[idx][: field.shape[0], : field.shape[1]].copy_(field)


class FakeNcclDistBackend(FakeTensorDistBackend):
    def __init__(self, payloads_by_rank):
        super().__init__(payloads_by_rank)
        self.created_groups = []

    def get_backend(self):
        return "nccl"

    def new_group(self, backend):
        self.created_groups.append(backend)
        return self


class FakeAllToAllDistBackend(FakeDistBackend):
    def __init__(self, payloads_by_rank, *, current_rank: int, backend_name: str = "gloo"):
        super().__init__(payloads_by_rank)
        self.current_rank = int(current_rank)
        self.backend_name = str(backend_name)
        self.created_groups = []
        self._pending_role = None
        self._pending_counts = None

    def get_backend(self):
        return self.backend_name

    def new_group(self, backend):
        self.created_groups.append(backend)
        return self

    def all_to_all_single(
        self,
        output_tensor,
        input_tensor,
        output_split_sizes=None,
        input_split_sizes=None,
        group=None,
    ):
        self.calls.append(("all_to_all_single", tuple(input_tensor.shape), str(input_tensor.dtype), group))
        local_payloads = self._payloads_by_rank[self.current_rank]
        if input_tensor.dtype == torch.int64 and input_tensor.dim() == 1 and input_tensor.numel() == len(local_payloads):
            # counts exchange
            recv_counts = []
            for source_rank in range(len(local_payloads)):
                payload = local_payloads[source_rank]
                if isinstance(payload, LookupIdsPayload):
                    recv_counts.append(int(payload.source_ranks.numel()))
                elif isinstance(payload, LookupEmbeddingResponsePayload):
                    recv_counts.append(int(payload.requestor_ranks.numel()))
                else:
                    recv_counts.append(int(payload.source_ranks.numel()))
            output_tensor.copy_(torch.tensor(recv_counts, dtype=torch.int64))
            self._pending_counts = recv_counts
            self._pending_role = None
            return

        role = None
        if input_tensor.dtype == torch.int64 and input_tensor.dim() == 1:
            if input_tensor.numel() % 4 == 0 and any(
                isinstance(payload, SparseGradPayload) for payload in local_payloads
            ):
                role = "packed_sparse"
            elif input_tensor.numel() % 3 == 0 and any(
                isinstance(payload, LookupIdsPayload) for payload in local_payloads
            ):
                role = "packed_lookup"
        if role is None and input_tensor.dtype == torch.float32 and input_tensor.dim() == 1:
            if any(isinstance(payload, SparseGradPayload) for payload in local_payloads):
                total_grad_numel = sum(int(payload.grads.numel()) for payload in local_payloads if isinstance(payload, SparseGradPayload))
                if total_grad_numel == input_tensor.numel():
                    role = "grad"
            elif any(isinstance(payload, LookupEmbeddingResponsePayload) for payload in local_payloads):
                total_emb_numel = sum(
                    int(payload.embeddings.numel())
                    for payload in local_payloads
                    if isinstance(payload, LookupEmbeddingResponsePayload)
                )
                if total_emb_numel == input_tensor.numel():
                    role = "emb"
        for payload in local_payloads:
            if role is not None:
                break
            candidates = []
            if isinstance(payload, LookupIdsPayload):
                candidates = [
                    ("source", payload.source_ranks),
                    ("row", payload.row_positions),
                    ("fused", payload.fused_ids),
                    (
                        "packed_lookup",
                        torch.stack(
                            [payload.source_ranks, payload.row_positions, payload.fused_ids],
                            dim=1,
                        ).view(-1),
                    ),
                ]
            elif isinstance(payload, LookupEmbeddingResponsePayload):
                candidates = [("requestor", payload.requestor_ranks), ("row", payload.row_positions), ("emb", payload.embeddings.view(-1))]
            else:
                candidates = [
                    ("dest", payload.destination_ranks),
                    ("source", payload.source_ranks),
                    ("row", payload.row_positions),
                    ("fused", payload.fused_ids),
                    (
                        "packed_sparse",
                        torch.stack(
                            [
                                payload.destination_ranks,
                                payload.source_ranks,
                                payload.row_positions,
                                payload.fused_ids,
                            ],
                            dim=1,
                        ).view(-1),
                    ),
                    ("grad", payload.grads.view(-1)),
                ]
            for candidate_role, candidate in candidates:
                if candidate.dtype == input_tensor.dtype and candidate.numel() == input_tensor.numel():
                    role = candidate_role
                    break
            if role is not None:
                break
        if role is None:
            raise AssertionError("unable to infer all_to_all field role")

        flat_parts = []
        for payload in local_payloads:
            if isinstance(payload, LookupIdsPayload):
                value = {
                    "source": payload.source_ranks,
                    "row": payload.row_positions,
                    "fused": payload.fused_ids,
                    "packed_lookup": torch.stack(
                        [payload.source_ranks, payload.row_positions, payload.fused_ids],
                        dim=1,
                    ).view(-1),
                }[role]
            elif isinstance(payload, LookupEmbeddingResponsePayload):
                value = {
                    "requestor": payload.requestor_ranks,
                    "row": payload.row_positions,
                    "emb": payload.embeddings.view(-1),
                }[role]
            else:
                value = {
                    "dest": payload.destination_ranks,
                    "source": payload.source_ranks,
                    "row": payload.row_positions,
                    "fused": payload.fused_ids,
                    "packed_sparse": torch.stack(
                        [
                            payload.destination_ranks,
                            payload.source_ranks,
                            payload.row_positions,
                            payload.fused_ids,
                        ],
                        dim=1,
                    ).view(-1),
                    "grad": payload.grads.view(-1),
                }[role]
            flat_parts.append(value.to(dtype=input_tensor.dtype))
        flat = torch.cat(flat_parts) if flat_parts else torch.empty((0,), dtype=input_tensor.dtype)
        output_tensor.copy_(flat.view_as(output_tensor))


class TestSingleNodeExchange(unittest.TestCase):
    def setUp(self):
        exchange_module._CPU_EXCHANGE_GROUP = None

    def _assert_grouping_and_permutation_helpers(self, device: torch.device) -> None:
        destination_ranks = torch.tensor([2, 0, 1, 2, 1, 0], dtype=torch.int64, device=device)
        send_counts, grouped_indices = exchange_module._group_row_indices_by_destination(
            destination_ranks,
            world_size=4,
        )

        self.assertEqual(send_counts, [2, 2, 2, 0])
        self.assertIsInstance(grouped_indices, torch.Tensor)
        self.assertEqual(grouped_indices.dtype, torch.int64)
        self.assertEqual(grouped_indices.device, destination_ranks.device)
        self.assertTrue(
            torch.equal(
                grouped_indices.cpu(),
                torch.tensor([1, 5, 2, 4, 0, 3], dtype=torch.int64),
            )
        )

        vector = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64, device=device)
        matrix = torch.tensor(
            [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5], [5.0, 5.5]],
            dtype=torch.float32,
            device=device,
        )
        int_matrix = torch.tensor(
            [[20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31]],
            dtype=torch.int64,
            device=device,
        )

        self.assertTrue(
            torch.equal(
                exchange_module._permute_vector_rows(vector, grouped_indices).cpu(),
                torch.tensor([11, 15, 12, 14, 10, 13], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                exchange_module._permute_matrix_rows(matrix, grouped_indices).cpu(),
                torch.tensor(
                    [[1.0, 1.5], [5.0, 5.5], [2.0, 2.5], [4.0, 4.5], [0.0, 0.5], [3.0, 3.5]],
                    dtype=torch.float32,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                exchange_module._permute_int_matrix_rows(int_matrix, grouped_indices).cpu(),
                torch.tensor(
                    [[22, 23], [30, 31], [24, 25], [28, 29], [20, 21], [26, 27]],
                    dtype=torch.int64,
                ),
            )
        )

    def test_exchange_lookup_ids_world_size_one_is_noop(self):
        payload = LookupIdsPayload(
            rank=0,
            destination_ranks=torch.tensor([0, 0], dtype=torch.int64),
            source_ranks=torch.tensor([0, 0], dtype=torch.int64),
            row_positions=torch.tensor([2, 5], dtype=torch.int64),
            fused_ids=torch.tensor([11, 17], dtype=torch.int64),
        )

        exchanged = exchange_lookup_ids(payload, world_size=1, backend=None)

        self.assertEqual(len(exchanged), 1)
        self.assertEqual(exchanged[0].rank, payload.rank)
        self.assertTrue(torch.equal(exchanged[0].source_ranks, payload.source_ranks))
        self.assertTrue(torch.equal(exchanged[0].row_positions, payload.row_positions))
        self.assertTrue(torch.equal(exchanged[0].fused_ids, payload.fused_ids))

    def test_exchange_lookup_ids_preserves_rank_local_metadata(self):
        local_payload = LookupIdsPayload(
            rank=1,
            destination_ranks=torch.tensor([0], dtype=torch.int64),
            source_ranks=torch.tensor([1], dtype=torch.int64),
            row_positions=torch.tensor([4], dtype=torch.int64),
            fused_ids=torch.tensor([13], dtype=torch.int64),
        )
        backend = FakeDistBackend(
            {
                1: [
                    LookupIdsPayload(
                        rank=0,
                        destination_ranks=torch.tensor([1, 1], dtype=torch.int64),
                        source_ranks=torch.tensor([0, 0], dtype=torch.int64),
                        row_positions=torch.tensor([1, 3], dtype=torch.int64),
                        fused_ids=torch.tensor([8, 12], dtype=torch.int64),
                    ),
                    local_payload,
                    LookupIdsPayload(
                        rank=2,
                        destination_ranks=torch.tensor([1, 1, 1], dtype=torch.int64),
                        source_ranks=torch.tensor([2, 2, 2], dtype=torch.int64),
                        row_positions=torch.tensor([0, 2, 6], dtype=torch.int64),
                        fused_ids=torch.tensor([5, 9, 14], dtype=torch.int64),
                    ),
                ]
            }
        )

        exchanged = exchange_lookup_ids(local_payload, world_size=3, backend=backend)

        self.assertEqual([payload.rank for payload in exchanged], [0, 1, 2])
        self.assertTrue(torch.equal(exchanged[0].row_positions, torch.tensor([1, 3], dtype=torch.int64)))
        self.assertTrue(torch.equal(exchanged[2].fused_ids, torch.tensor([5, 9, 14], dtype=torch.int64)))
        self.assertEqual(backend.calls, ["LookupIdsPayload"])

    def test_exchange_lookup_ids_uses_tensor_collective_fast_path_when_available(self):
        local_payload = LookupIdsPayload(
            rank=1,
            destination_ranks=torch.tensor([0], dtype=torch.int64),
            source_ranks=torch.tensor([1], dtype=torch.int64),
            row_positions=torch.tensor([4], dtype=torch.int64),
            fused_ids=torch.tensor([13], dtype=torch.int64),
        )
        backend = FakeTensorDistBackend(
            {
                1: [
                    LookupIdsPayload(
                        rank=0,
                        destination_ranks=torch.tensor([1, 1], dtype=torch.int64),
                        source_ranks=torch.tensor([0, 0], dtype=torch.int64),
                        row_positions=torch.tensor([1, 3], dtype=torch.int64),
                        fused_ids=torch.tensor([8, 12], dtype=torch.int64),
                    ),
                    local_payload,
                    LookupIdsPayload(
                        rank=2,
                        destination_ranks=torch.tensor([1], dtype=torch.int64),
                        source_ranks=torch.tensor([2], dtype=torch.int64),
                        row_positions=torch.tensor([6], dtype=torch.int64),
                        fused_ids=torch.tensor([14], dtype=torch.int64),
                    ),
                ]
            }
        )

        exchanged = exchange_lookup_ids(local_payload, world_size=3, backend=backend)

        self.assertEqual([payload.rank for payload in exchanged], [0, 1, 2])
        self.assertTrue(any(call[0] == "tensor" for call in backend.calls if isinstance(call, tuple)))

    def test_reassemble_lookup_embedding_responses_restores_request_order(self):
        responses = [
            LookupEmbeddingResponsePayload(
                rank=0,
                requestor_ranks=torch.tensor([0, 0], dtype=torch.int64),
                row_positions=torch.tensor([3, 0], dtype=torch.int64),
                embeddings=torch.tensor([[30.0, 31.0], [0.0, 1.0]], dtype=torch.float32),
            ),
            LookupEmbeddingResponsePayload(
                rank=2,
                requestor_ranks=torch.tensor([0], dtype=torch.int64),
                row_positions=torch.tensor([2], dtype=torch.int64),
                embeddings=torch.tensor([[20.0, 21.0]], dtype=torch.float32),
            ),
            LookupEmbeddingResponsePayload(
                rank=1,
                requestor_ranks=torch.tensor([0], dtype=torch.int64),
                row_positions=torch.tensor([1], dtype=torch.int64),
                embeddings=torch.tensor([[10.0, 11.0]], dtype=torch.float32),
            ),
        ]

        rebuilt = reassemble_lookup_embedding_responses(
            responses,
            requestor_rank=0,
            total_rows=4,
        )

        expected = torch.tensor(
            [[0.0, 1.0], [10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(rebuilt, expected))

    def test_exchange_lookup_embedding_responses_gathers_payloads(self):
        local_payload = LookupEmbeddingResponsePayload(
            rank=1,
            requestor_ranks=torch.tensor([0], dtype=torch.int64),
            row_positions=torch.tensor([2], dtype=torch.int64),
            embeddings=torch.tensor([[2.0, 3.0]], dtype=torch.float32),
        )
        backend = FakeDistBackend(
            {
                1: [
                    LookupEmbeddingResponsePayload(
                        rank=0,
                        requestor_ranks=torch.tensor([0], dtype=torch.int64),
                        row_positions=torch.tensor([1], dtype=torch.int64),
                        embeddings=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
                    ),
                    local_payload,
                ]
            }
        )

        exchanged = exchange_lookup_embedding_responses(local_payload, world_size=2, backend=backend)

        self.assertEqual([payload.rank for payload in exchanged], [0, 1])
        self.assertTrue(torch.equal(exchanged[1].embeddings, local_payload.embeddings))

    def test_exchange_sparse_grads_uses_tensor_collective_fast_path_when_available(self):
        local_payload = SparseGradPayload(
            rank=1,
            destination_ranks=torch.tensor([0], dtype=torch.int64),
            source_ranks=torch.tensor([1], dtype=torch.int64),
            row_positions=torch.tensor([0], dtype=torch.int64),
            fused_ids=torch.tensor([5], dtype=torch.int64),
            grads=torch.tensor([[0.5, 1.5]], dtype=torch.float32),
        )
        backend = FakeTensorDistBackend(
            {
                1: [
                    SparseGradPayload(
                        rank=0,
                        destination_ranks=torch.tensor([1], dtype=torch.int64),
                        source_ranks=torch.tensor([0], dtype=torch.int64),
                        row_positions=torch.tensor([3], dtype=torch.int64),
                        fused_ids=torch.tensor([4], dtype=torch.int64),
                        grads=torch.tensor([[4.5, 5.5]], dtype=torch.float32),
                    ),
                    local_payload,
                ]
            }
        )

        exchanged = exchange_sparse_grads(local_payload, world_size=2, backend=backend)

        self.assertEqual(len(exchanged), 2)
        self.assertTrue(torch.equal(exchanged[1].grads, local_payload.grads))

    def test_exchange_uses_cpu_gloo_group_when_default_backend_is_nccl(self):
        local_payload = LookupIdsPayload(
            rank=0,
            destination_ranks=torch.tensor([0], dtype=torch.int64),
            source_ranks=torch.tensor([0], dtype=torch.int64),
            row_positions=torch.tensor([1], dtype=torch.int64),
            fused_ids=torch.tensor([9], dtype=torch.int64),
        )
        backend = FakeNcclDistBackend(
            {
                0: [
                    local_payload,
                    LookupIdsPayload(
                        rank=1,
                        destination_ranks=torch.tensor([0], dtype=torch.int64),
                        source_ranks=torch.tensor([1], dtype=torch.int64),
                        row_positions=torch.tensor([2], dtype=torch.int64),
                        fused_ids=torch.tensor([10], dtype=torch.int64),
                    ),
                ]
            }
        )

        exchanged = exchange_lookup_ids(local_payload, world_size=2, backend=backend)

        self.assertEqual([payload.rank for payload in exchanged], [0, 1])
        self.assertEqual(backend.created_groups, ["gloo"])
        self.assertTrue(any(call[-1] is backend for call in backend.calls if isinstance(call, tuple)))

    def test_exchange_sparse_grads_preserves_gradient_payload(self):
        local_payload = SparseGradPayload(
            rank=2,
            destination_ranks=torch.tensor([0, 1], dtype=torch.int64),
            source_ranks=torch.tensor([1, 1], dtype=torch.int64),
            row_positions=torch.tensor([0, 4], dtype=torch.int64),
            fused_ids=torch.tensor([5, 7], dtype=torch.int64),
            grads=torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32),
        )
        backend = FakeDistBackend(
            {
                2: [
                    SparseGradPayload(
                        rank=0,
                        destination_ranks=torch.tensor([], dtype=torch.int64),
                        source_ranks=torch.tensor([], dtype=torch.int64),
                        row_positions=torch.tensor([], dtype=torch.int64),
                        fused_ids=torch.tensor([], dtype=torch.int64),
                        grads=torch.empty((0, 2), dtype=torch.float32),
                    ),
                    SparseGradPayload(
                        rank=1,
                        destination_ranks=torch.tensor([2], dtype=torch.int64),
                        source_ranks=torch.tensor([1], dtype=torch.int64),
                        row_positions=torch.tensor([3], dtype=torch.int64),
                        fused_ids=torch.tensor([4], dtype=torch.int64),
                        grads=torch.tensor([[4.5, 5.5]], dtype=torch.float32),
                    ),
                    local_payload,
                ]
            }
        )

        exchanged = exchange_sparse_grads(local_payload, world_size=3, backend=backend)

        self.assertEqual(len(exchanged), 3)
        self.assertTrue(torch.equal(exchanged[2].grads, local_payload.grads))
        self.assertTrue(torch.equal(exchanged[2].source_ranks, torch.tensor([1, 1], dtype=torch.int64)))

    def test_reassemble_lookup_embedding_responses_preserves_embedding_device(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        responses = [
            LookupEmbeddingResponsePayload(
                rank=0,
                requestor_ranks=torch.tensor([0], dtype=torch.int64),
                row_positions=torch.tensor([0], dtype=torch.int64),
                embeddings=torch.tensor([[1.0, 2.0]], dtype=torch.float32, device=device),
            ),
            LookupEmbeddingResponsePayload(
                rank=1,
                requestor_ranks=torch.tensor([0], dtype=torch.int64),
                row_positions=torch.tensor([1], dtype=torch.int64),
                embeddings=torch.tensor([[3.0, 4.0]], dtype=torch.float32, device=device),
            ),
        ]

        rebuilt = reassemble_lookup_embedding_responses(
            responses,
            requestor_rank=0,
            total_rows=2,
        )

        self.assertEqual(rebuilt.device, device)
        self.assertTrue(
            torch.equal(
                rebuilt.cpu(),
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            )
        )

    def test_exchange_lookup_ids_rejects_rank_slot_mismatch(self):
        local_payload = LookupIdsPayload(
            rank=1,
            destination_ranks=torch.tensor([0], dtype=torch.int64),
            source_ranks=torch.tensor([1], dtype=torch.int64),
            row_positions=torch.tensor([4], dtype=torch.int64),
            fused_ids=torch.tensor([13], dtype=torch.int64),
        )
        backend = FakeDistBackend(
            {
                1: [
                    LookupIdsPayload(
                        rank=1,
                        destination_ranks=torch.tensor([1], dtype=torch.int64),
                        source_ranks=torch.tensor([0], dtype=torch.int64),
                        row_positions=torch.tensor([0], dtype=torch.int64),
                        fused_ids=torch.tensor([5], dtype=torch.int64),
                    ),
                    local_payload,
                ]
            }
        )

        with self.assertRaisesRegex(RuntimeError, "gathered payload rank metadata mismatch"):
            exchange_lookup_ids(local_payload, world_size=2, backend=backend)

    def test_exchange_lookup_ids_uses_all_to_all_single_when_available(self):
        local_payload = LookupIdsPayload(
            rank=0,
            destination_ranks=torch.tensor([0, 1], dtype=torch.int64),
            source_ranks=torch.tensor([0, 0], dtype=torch.int64),
            row_positions=torch.tensor([3, 4], dtype=torch.int64),
            fused_ids=torch.tensor([13, 17], dtype=torch.int64),
        )
        backend = FakeAllToAllDistBackend(
            {
                0: [
                    LookupIdsPayload(
                        rank=0,
                        destination_ranks=torch.tensor([0], dtype=torch.int64),
                        source_ranks=torch.tensor([0], dtype=torch.int64),
                        row_positions=torch.tensor([3], dtype=torch.int64),
                        fused_ids=torch.tensor([13], dtype=torch.int64),
                    ),
                    LookupIdsPayload(
                        rank=1,
                        destination_ranks=torch.tensor([0], dtype=torch.int64),
                        source_ranks=torch.tensor([1], dtype=torch.int64),
                        row_positions=torch.tensor([9], dtype=torch.int64),
                        fused_ids=torch.tensor([21], dtype=torch.int64),
                    ),
                ]
            },
            current_rank=0,
        )

        exchanged = exchange_lookup_ids(local_payload, world_size=2, backend=backend)

        self.assertEqual([payload.rank for payload in exchanged], [0, 1])
        self.assertTrue(any(call[0] == "all_to_all_single" for call in backend.calls if isinstance(call, tuple)))

    def test_group_row_indices_helpers_keep_tensor_indices_on_cpu(self):
        self._assert_grouping_and_permutation_helpers(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for NCCL device-preservation coverage")
    def test_exchange_lookup_ids_keeps_cuda_payload_fields_on_nccl_all_to_all(self):
        device = torch.device("cuda", 0)
        local_payload = LookupIdsPayload(
            rank=0,
            destination_ranks=torch.tensor([0, 1], dtype=torch.int64, device=device),
            source_ranks=torch.tensor([0, 0], dtype=torch.int64, device=device),
            row_positions=torch.tensor([3, 4], dtype=torch.int64, device=device),
            fused_ids=torch.tensor([13, 17], dtype=torch.int64, device=device),
        )
        backend = FakeAllToAllDistBackend(
            {
                0: [
                    LookupIdsPayload(
                        rank=0,
                        destination_ranks=torch.tensor([0], dtype=torch.int64, device=device),
                        source_ranks=torch.tensor([0], dtype=torch.int64, device=device),
                        row_positions=torch.tensor([3], dtype=torch.int64, device=device),
                        fused_ids=torch.tensor([13], dtype=torch.int64, device=device),
                    ),
                    LookupIdsPayload(
                        rank=1,
                        destination_ranks=torch.tensor([0], dtype=torch.int64, device=device),
                        source_ranks=torch.tensor([1], dtype=torch.int64, device=device),
                        row_positions=torch.tensor([9], dtype=torch.int64, device=device),
                        fused_ids=torch.tensor([21], dtype=torch.int64, device=device),
                    ),
                ]
            },
            current_rank=0,
            backend_name="nccl",
        )

        exchanged = exchange_lookup_ids(local_payload, world_size=2, backend=backend)

        self.assertEqual(backend.created_groups, [])
        for payload in exchanged:
            self.assertEqual(payload.destination_ranks.device.type, "cuda")
            self.assertEqual(payload.source_ranks.device.type, "cuda")
            self.assertEqual(payload.row_positions.device.type, "cuda")
            self.assertEqual(payload.fused_ids.device.type, "cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for grouped-index device coverage")
    def test_group_row_indices_helpers_keep_tensor_indices_on_cuda(self):
        self._assert_grouping_and_permutation_helpers(torch.device("cuda", 0))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for NCCL device-preservation coverage")
    def test_exchange_sparse_grads_keeps_cuda_payload_fields_on_nccl_all_to_all(self):
        device = torch.device("cuda", 0)
        local_payload = SparseGradPayload(
            rank=0,
            destination_ranks=torch.tensor([0, 1], dtype=torch.int64, device=device),
            source_ranks=torch.tensor([0, 0], dtype=torch.int64, device=device),
            row_positions=torch.tensor([3, 4], dtype=torch.int64, device=device),
            fused_ids=torch.tensor([13, 17], dtype=torch.int64, device=device),
            grads=torch.tensor([[1.0, 1.5], [2.0, 2.5]], dtype=torch.float32, device=device),
        )
        backend = FakeAllToAllDistBackend(
            {
                0: [
                    SparseGradPayload(
                        rank=0,
                        destination_ranks=torch.tensor([0], dtype=torch.int64, device=device),
                        source_ranks=torch.tensor([0], dtype=torch.int64, device=device),
                        row_positions=torch.tensor([3], dtype=torch.int64, device=device),
                        fused_ids=torch.tensor([13], dtype=torch.int64, device=device),
                        grads=torch.tensor([[1.0, 1.5]], dtype=torch.float32, device=device),
                    ),
                    SparseGradPayload(
                        rank=1,
                        destination_ranks=torch.tensor([0], dtype=torch.int64, device=device),
                        source_ranks=torch.tensor([1], dtype=torch.int64, device=device),
                        row_positions=torch.tensor([9], dtype=torch.int64, device=device),
                        fused_ids=torch.tensor([21], dtype=torch.int64, device=device),
                        grads=torch.tensor([[3.0, 3.5]], dtype=torch.float32, device=device),
                    ),
                ]
            },
            current_rank=0,
            backend_name="nccl",
        )

        exchanged = exchange_sparse_grads(local_payload, world_size=2, backend=backend)

        self.assertEqual(backend.created_groups, [])
        for payload in exchanged:
            self.assertEqual(payload.destination_ranks.device.type, "cuda")
            self.assertEqual(payload.source_ranks.device.type, "cuda")
            self.assertEqual(payload.row_positions.device.type, "cuda")
            self.assertEqual(payload.fused_ids.device.type, "cuda")
            self.assertEqual(payload.grads.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
