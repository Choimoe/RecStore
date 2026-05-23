# RecStore Agent Guidelines

These instructions apply repository-wide unless the current conversation gives a
more specific instruction.

## Language

- Write `AGENTS.md` and agent-facing operating instructions in English.
- Reply to the user in Chinese by default.
- Write project documentation in Chinese by default unless requested otherwise.
- Write code comments in English by default.

## Task-Specific Guides

Read the relevant guide before doing specialized work:

- Performance comparisons and benchmark reports: `docs/agent/perf.md`

## Git Rules

- Default commit messages must be English Conventional Commits, for example
  `feat(scope): ...`, `fix(scope): ...`, `docs(scope): ...`.
- Do not amend commits unless explicitly requested.
- Never use destructive commands such as `git reset --hard` or
  `git checkout --` unless explicitly requested.
- Assume the worktree may be dirty. Do not revert unrelated user changes.
- Do not commit transient AI planning files or scratch artifacts such as
  `docs/superpowers/specs/*`.

## Development Workflow

For feature work or non-trivial bug fixes:

1. Understand the local context first.
2. Follow existing design or propose a small design when needed.
3. Implement in small, reviewable increments.
4. Verify with the narrowest useful tests first, then broader checks when risk
   or blast radius requires it.

Do not claim completion before running verification that actually exercises the
changed behavior.

## Architecture Boundaries

Keep these boundaries explicit:

- storage and server behavior
- Python client protocol and semantics
- model integration glue
- training-loop scheduling and optimization logic

Prefer explicit context passing, obvious synchronization points, and loud
failures when invariants are violated. Avoid hidden shared mutable state,
misleading async APIs, and broad refactors that obscure behavior changes.

## Review Focus

Prioritize correctness before performance. Pay special attention to:

- sparse update visibility across training steps
- prefetch and read-after-write ordering
- tensor device, dtype, and shape mismatches
- fallback correctness when optimized paths are unavailable
- background thread lifecycle, shutdown, and exception propagation
- consistency between Python wrappers and backend behavior

## Coding Rules

- Follow existing repository patterns before introducing abstractions.
- Prefer readable, local changes over clever or broad refactors.
- Use ASCII by default unless the file already requires non-ASCII content.
- Add comments only for non-obvious intent or invariants.
- In Python, make submission, wait, and consumption semantics explicit.
- In C++, preserve surrounding ownership and synchronization style.

## Verified Lessons

- Do not trust README claims without checking the actual code path.
- Distributed RecStore routing must follow `distributed_client`; treat
  `hash_method`, `num_shards`, and `servers` as separate fields.
- Do not assume `shard == sorted_index`; route by explicit shard id.
- Python wrappers that recreate backend routing must match backend semantics or
  fail loudly.
- Async-looking APIs are not automatically safe; verify handle uniqueness and
  visibility semantics.
- For correctness, stable `prefetch + immediate wait` is better than a
  misleading async path.
- For local benchmark bring-up, first get a stable closed loop. If
  `R2ShmMalloc` makes `ps_server` unstable, `PersistLoopShmMalloc` is acceptable
  as a temporary baseline, but state it in the report.
- A result comparing `TorchRec-Local-HBM` with `RecStore-Local-RPC` is a lane
  observation, not architecture-level proof.

## Testing

- Run the most relevant targeted tests for the changed area.
- Add tests when behavior is not already covered.
- If tests cannot run in the current environment, say why.

Useful verification layers:

- Python unit tests under `src/python/pytorch/recstore/unittest`
- model-zoo integration checks under `model_zoo/torchrec_dlrm`
- `model_zoo/rs_demo` smoke and benchmark runs
- compiled targets in `build/`
- server/client smoke tests against `ps_server`

## PyTorch Client Verification

When asked to validate baseline PyTorch client operability:

1. Confirm `build/` exists.
2. Run `make -j` inside `build/`.
3. Start `./build/bin/ps_server --config_path ./recstore_config.json`.
4. Confirm shards listen on the ports in `recstore_config.json`.
5. Run `ctest -R pytorch_client_test -VV` inside `build/`.
6. Stop the manually started server.

## Editing Safety

- Read relevant code before editing.
- Never overwrite or revert user changes just to simplify your patch.
- Work with unexpected changes unless they directly block correctness.
- Ask only when conflicting changes make the right resolution unclear.
- Keep patches scoped to the task.
- Do not present hypothetical fixes as completed work.
- Do not say tests pass unless you ran them.
