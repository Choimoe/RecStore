from __future__ import annotations

import dataclasses
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..config import RunConfig, dump_run_config, ensure_shared_dir
from ..runtime.worker_common import build_worker_env, merge_rank_outputs


class BenchmarkRunner(ABC):
    """Benchmark backend with shared torchrun plumbing.

    Subclasses own model construction and the training loop. Spawning
    workers, the launcher argv shape, worker env, and rank CSV collection
    live here so the three backends cannot drift apart.
    """

    backend = ""
    rank_dir_name = ""

    @abstractmethod
    def run(self, repo_root: Path, cfg: RunConfig) -> dict:
        raise NotImplementedError

    # -- layout ---------------------------------------------------------

    def _rank_output_dir(self, cfg: RunConfig) -> Path:
        return Path(cfg.output_root) / "outputs" / cfg.run_id / self.rank_dir_name

    def _main_csv(self, cfg: RunConfig) -> Path:
        raise NotImplementedError

    # -- backend hooks --------------------------------------------------

    def _worker_argv(self, cfg: RunConfig) -> list[str]:
        """Launcher args for backends that do not use ``--run-config-json``."""
        return []

    def _prepare_distributed(self, cfg: RunConfig, rank_dir: Path) -> None:
        pass

    def _worker_env(self, rank_dir: Path) -> dict[str, str]:
        return build_worker_env(self.backend, rank_dir)

    def _after_torchrun(
        self, cfg: RunConfig, rank_dir: Path, result: subprocess.CompletedProcess
    ) -> None:
        pass

    # -- shared launch --------------------------------------------------

    def _build_torchrun_cmd(
        self, repo_root: Path, cfg: RunConfig, config_json: Path | None = None
    ) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes",
            str(cfg.nnodes),
            "--node_rank",
            str(cfg.node_rank),
            "--nproc_per_node",
            str(cfg.nproc_per_node),
            "--rdzv_backend",
            str(cfg.rdzv_backend),
            "--rdzv_endpoint",
            f"{cfg.master_addr}:{cfg.master_port}",
            "--rdzv_id",
            str(cfg.rdzv_id),
            "--tee",
            "3",
            str(repo_root / "model_zoo/rs_demo/run_mock_stress.py"),
        ]
        if config_json is None:
            cmd.extend(self._worker_argv(cfg))
        else:
            cmd.extend(["--run-config-json", str(config_json)])
        return cmd

    def _run_distributed(
        self, repo_root: Path, cfg: RunConfig, *, config_json: bool = True
    ) -> dict[str, Any]:
        rank_dir = self._rank_output_dir(cfg)
        ensure_shared_dir(rank_dir)
        self._prepare_distributed(cfg, rank_dir)

        if config_json:
            # The whole resolved config is handed to each worker as JSON, so
            # the launcher argv stays tiny and never drifts from the schema.
            worker_cfg = dataclasses.replace(cfg, start_server=False)
            cmd = self._build_torchrun_cmd(
                repo_root, cfg, dump_run_config(worker_cfg, rank_dir / "worker_config.json")
            )
        else:
            cmd = self._build_torchrun_cmd(repo_root, cfg)

        env = self._worker_env(rank_dir)
        result = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        self._after_torchrun(cfg, rank_dir, result)
        if result.returncode != 0:
            raise RuntimeError(
                f"{self.backend} torchrun worker failed\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        return {"backend": self.backend, "rows": self._merge_rank_csvs(cfg, rank_dir)}

    def _merge_rank_csvs(
        self, cfg: RunConfig, rank_dir: Path
    ) -> list[dict[str, Any]]:
        world_size = cfg.nnodes * cfg.nproc_per_node
        rank_csvs = [rank_dir / f"rank{rank}.csv" for rank in range(world_size)]
        # Without a shared filesystem each host can only see its own rank
        # CSVs; verify the local ranks and merge everything only when all
        # ranks are visible. The e2e driver collects remote rank CSVs and
        # produces the merged main csv itself.
        local_ranks = range(
            cfg.node_rank * cfg.nproc_per_node,
            (cfg.node_rank + 1) * cfg.nproc_per_node,
        )
        missing_local = [
            str(rank_dir / f"rank{rank}.csv")
            for rank in local_ranks
            if not (rank_dir / f"rank{rank}.csv").exists()
        ]
        if missing_local:
            raise RuntimeError(f"missing local rank csv outputs: {missing_local}")
        missing = [str(path) for path in rank_csvs if not path.exists()]
        if missing:
            print(
                "[rs_demo] remote rank csvs not visible on this host "
                f"(no shared filesystem?); skipping merge: {missing}"
            )
            return []
        return merge_rank_outputs(rank_csvs, self._main_csv(cfg))
