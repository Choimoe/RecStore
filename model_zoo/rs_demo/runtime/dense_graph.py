"""FP32 DLRM forward/backward capture with SGD and communication outside."""

import time

import torch


class CapturedDlrmDense:
    def __init__(
        self, module, criterion, gradient_buffer, *, batch_sizes,
        num_features, embedding_dim, dense_features=13,
        resident_values=None, payload_capacities=None,
    ):
        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            raise ValueError("Capture the unwrapped DLRM module; communicate outside capture")
        self.module = module
        self.gradient_buffer = gradient_buffer
        self.params = gradient_buffer.params
        module_params = [p for p in module.parameters() if p.requires_grad]
        if len(module_params) != len(self.params) or any(
            p is not q for p, q in zip(module_params, self.params)
        ):
            raise ValueError("The gradient buffer must belong to the captured module")
        if not self.params or any(
            p.device.type != "cuda" or p.dtype != torch.float32
            or not p.is_contiguous() or p.device != self.params[0].device
            for p in self.params
        ):
            raise ValueError("DLRM capture requires contiguous FP32 CUDA parameters")
        self.device = self.params[0].device
        batch_sizes = tuple(dict.fromkeys(batch_sizes))
        self.resident_values = resident_values
        if (resident_values is None) != (payload_capacities is None):
            raise ValueError("Planned capture requires resident values and payload capacities together")
        if resident_values is not None:
            if (
                resident_values.ndim != 2 or resident_values.shape[0] == 0
                or resident_values.shape[1] != embedding_dim
                or resident_values.dtype != torch.float32 or resident_values.device != self.device
                or resident_values.requires_grad or not resident_values.is_contiguous()
            ):
                raise ValueError("Planned capture requires contiguous resident FP32 CUDA rows")
            if set(payload_capacities) != set(batch_sizes) or any(
                not isinstance(count, int) or count <= 0 for count in payload_capacities.values()
            ):
                raise ValueError("Every captured batch size requires a positive payload capacity")
        self.flat = gradient_buffer.flat
        self.views = gradient_buffer.views
        self.graphs = {}
        self.stream = torch.cuda.Stream(device=self.device)
        initial_state = {name: value.clone() for name, value in module.state_dict().items()}
        parameter_pointers = [p.data_ptr() for p in self.params]
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(self.device)
        start = time.perf_counter()
        for batch_size in batch_sizes:
            if batch_size <= 0:
                raise ValueError("Captured batch sizes must be positive")
            dense = torch.zeros((batch_size, dense_features), device=self.device, dtype=torch.float32)
            sparse = torch.zeros(
                (batch_size, num_features, embedding_dim), device=self.device,
                dtype=torch.float32, requires_grad=True,
            )
            labels = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)
            sparse.grad = torch.zeros_like(sparse)
            sparse_grad = sparse.grad
            lookup_rows = pooled_rows = payload = None
            if resident_values is not None:
                lookup_rows = torch.zeros(batch_size * num_features, device=self.device, dtype=torch.int64)
                pooled_rows = torch.zeros_like(lookup_rows)
                payload = torch.empty(
                    (payload_capacities[batch_size], embedding_dim), device=self.device, dtype=torch.float32,
                )

            def forward_backward():
                gradient_buffer.begin_step()
                sparse_grad.zero_()
                if resident_values is not None:
                    with torch.no_grad():
                        torch.index_select(resident_values, 0, lookup_rows, out=sparse.view(-1, embedding_dim))
                loss = criterion(module(dense, sparse), labels)
                loss.backward()
                if payload is not None:
                    payload.zero_()
                    payload.index_add_(0, pooled_rows, sparse_grad.reshape(-1, embedding_dim))
                return loss

            # Zero inputs preserve RNG state; warmup never runs an optimizer.
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.stream):
                for _ in range(3):
                    forward_backward()
            self.stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            # Separate pools permit arbitrary replay order across batch sizes.
            with torch.cuda.graph(graph, stream=self.stream):
                loss = forward_backward()
            if len(gradient_buffer.seen) != len(self.params):
                raise RuntimeError("Every dense parameter must participate in captured backward")
            if any(p.grad is not view for p, view in zip(self.params, self.views)):
                raise RuntimeError("Capture replaced a fixed dense gradient view")
            if sparse.grad is not sparse_grad:
                raise RuntimeError("Capture replaced a fixed sparse gradient tensor")
            self.graphs[batch_size] = (
                graph, dense, sparse, labels, loss, sparse_grad, lookup_rows, pooled_rows, payload,
            )
        if not self.graphs:
            raise ValueError("At least one batch size must be captured")
        torch.cuda.synchronize(self.device)
        self.capture_ms = (time.perf_counter() - start) * 1000
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, initial_state[name], rtol=0, atol=0)
        if parameter_pointers != [p.data_ptr() for p in self.params]:
            raise RuntimeError("Capture changed parameter storage")
        if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(
            cuda_rng, torch.cuda.get_rng_state(self.device)
        ):
            raise RuntimeError("DLRM capture changed RNG state")

    def __call__(self, dense, sparse, labels, *, lookup_rows=None, pooled_rows=None, payload_rows=None):
        if dense.ndim != 2 or dense.shape[0] not in self.graphs:
            raise ValueError("Input batch size has no captured DLRM graph")
        (
            graph, static_dense, static_sparse, static_labels, loss, sparse_grad,
            static_lookup, static_pooled, payload,
        ) = self.graphs[dense.shape[0]]
        inputs = [(dense, static_dense), (labels, static_labels)]
        if self.resident_values is None:
            if lookup_rows is not None or pooled_rows is not None or payload_rows is not None:
                raise ValueError("Planned maps require a graph captured with resident values")
            inputs.append((sparse, static_sparse))
        else:
            if sparse is not None or not isinstance(payload_rows, int) or not 0 < payload_rows <= payload.shape[0]:
                raise ValueError("Planned replay requires no sparse input and a payload count within capacity")
            for actual, expected in ((lookup_rows, static_lookup), (pooled_rows, static_pooled)):
                if (
                    actual is None or actual.shape != expected.shape
                    or actual.dtype != torch.int64 or actual.device != self.device
                ):
                    raise ValueError("Planned maps must match the captured int64 CUDA shape")
        for actual, expected in inputs:
            if actual.shape != expected.shape or actual.dtype != torch.float32 or actual.device != self.device:
                raise ValueError("Input shape, FP32 dtype, or device differs from the captured DLRM input")
        # Optimizer.zero_grad(set_to_none=True) may remove these bindings. Do
        # not call begin_step: replay runs captured zeroing, but no Python hooks.
        for param, view in zip(self.params, self.views):
            param.grad = view
        with torch.no_grad():
            for actual, expected in inputs:
                expected.copy_(actual)
            if self.resident_values is not None:
                static_lookup.copy_(lookup_rows)
                static_pooled.copy_(pooled_rows)
        graph.replay()
        # Both outputs are overwritten on the next replay of this batch size.
        return loss.detach(), sparse_grad if payload is None else payload[:payload_rows]
