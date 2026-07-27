import json
import os
import sys
from pathlib import Path
from typing import Optional

SRC_ROOT = Path(__file__).resolve().parents[5]
REPO_ROOT = Path(__file__).resolve().parents[6]
TEST_SCRIPTS_PATH = SRC_ROOT / "test" / "scripts"

for path in (REPO_ROOT, TEST_SCRIPTS_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.config.recstore_config_path import resolve_recstore_config_path


def resolve_repo_config_path(start_dir: Optional[Path | str] = None) -> str:
    return str(resolve_recstore_config_path(start_dir))


def resolve_ps_endpoint(config_path: Optional[Path | str] = None) -> tuple[str, int]:
    if config_path is None:
        config_path = resolve_repo_config_path()

    with open(config_path, "r") as f:
        config = json.load(f)

    host = "127.0.0.1"
    port = 15000

    client_config = config.get("client", {})
    if isinstance(client_config, dict):
        if client_config.get("host") is not None:
            host = str(client_config["host"])
        if client_config.get("port") is not None:
            return host, int(client_config["port"])

    cache_servers = config.get("cache_ps", {}).get("servers", [])
    if isinstance(cache_servers, list) and cache_servers:
        first_server = cache_servers[0]
        if isinstance(first_server, dict):
            if first_server.get("host") is not None:
                host = str(first_server["host"])
            if first_server.get("port") is not None:
                port = int(first_server["port"])

    return host, port


def configure_src_paths() -> None:
    # Prefer the canonical package roots used by EmbeddingBag (`recstore`,
    # `torchrec_kv`). Putting `src/` on the path alone yields a second package
    # identity (`python.pytorch.recstore`) and split KVClient singletons.
    pytorch_root = os.path.abspath(str(Path(__file__).resolve().parents[3]))
    for path in (pytorch_root, os.path.abspath(str(SRC_ROOT))):
        if path not in sys.path:
            sys.path.insert(0, path)

