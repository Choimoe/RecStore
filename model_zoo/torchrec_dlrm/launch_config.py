import ast
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Set

try:
    import gin  # type: ignore
except ImportError:
    gin = None


@dataclass(eq=True)
class SingleDayLaunchConfig:
    use_torchrec: bool = False
    processed_dataset_path: str = "./processed_day_0_data"
    batch_size: int = 1024
    learning_rate: float = 0.005
    epochs: int = 1
    enable_prefetch: bool = True
    prefetch_depth: int = 2
    fuse_emb_tables: bool = True
    fuse_k: int = 30
    trace_file: str = ""
    allow_tf32: bool = False
    embedding_storage: str = "hbm"
    log_dir: str = "runs"
    run_name: str = ""
    nproc_per_node: int = 1
    rdzv_backend: str = "c10d"
    rdzv_endpoint: str = "localhost"
    rdzv_id: str = ""


_CONFIG_FIELDS = {
    "use_torchrec",
    "processed_dataset_path",
    "batch_size",
    "learning_rate",
    "epochs",
    "enable_prefetch",
    "prefetch_depth",
    "fuse_emb_tables",
    "fuse_k",
    "trace_file",
    "allow_tf32",
    "embedding_storage",
    "log_dir",
    "run_name",
    "nproc_per_node",
    "rdzv_backend",
    "rdzv_endpoint",
    "rdzv_id",
}

_ARG_TO_CONFIG_KEY = {
    "batch_size": "batch_size",
    "learning_rate": "learning_rate",
    "epochs": "epochs",
    "in_memory_binary_criteo_path": "processed_dataset_path",
    "enable_prefetch": "enable_prefetch",
    "prefetch_depth": "prefetch_depth",
    "fuse_emb_tables": "fuse_emb_tables",
    "fuse_k": "fuse_k",
    "trace_file": "trace_file",
    "allow_tf32": "allow_tf32",
    "embedding_storage": "embedding_storage",
    "log_dir": "log_dir",
    "run_name": "run_name",
    "nproc_per_node": "nproc_per_node",
    "rdzv_backend": "rdzv_backend",
    "rdzv_endpoint": "rdzv_endpoint",
    "rdzv_id": "rdzv_id",
}

_CLI_OPTION_TO_CONFIG_KEY = {
    "--torchrec": "use_torchrec",
    "--custom": "use_torchrec",
    "--dataset-path": "processed_dataset_path",
    "--batch-size": "batch_size",
    "--learning-rate": "learning_rate",
    "--epochs": "epochs",
    "--enable-prefetch": "enable_prefetch",
    "--disable-prefetch": "enable_prefetch",
    "--no-prefetch": "enable_prefetch",
    "--prefetch-depth": "prefetch_depth",
    "--enable-fuse-emb": "fuse_emb_tables",
    "--disable-fuse-emb": "fuse_emb_tables",
    "--no-fuse-emb": "fuse_emb_tables",
    "--fuse-k": "fuse_k",
    "--trace-file": "trace_file",
    "--allow-tf32": "allow_tf32",
    "--embedding-storage": "embedding_storage",
    "--in_memory_binary_criteo_path": "processed_dataset_path",
    "--batch_size": "batch_size",
    "--learning_rate": "learning_rate",
    "--trace_file": "trace_file",
    "--prefetch_depth": "prefetch_depth",
    "--embedding_storage": "embedding_storage",
    "--log-dir": "log_dir",
    "--run-name": "run_name",
    "--nproc-per-node": "nproc_per_node",
    "--rdzv-backend": "rdzv_backend",
    "--rdzv-endpoint": "rdzv_endpoint",
    "--rdzv-id": "rdzv_id",
    "--log_dir": "log_dir",
    "--run_name": "run_name",
    "--nproc_per_node": "nproc_per_node",
    "--rdzv_backend": "rdzv_backend",
    "--rdzv_endpoint": "rdzv_endpoint",
    "--rdzv_id": "rdzv_id",
}


def _parse_literal(raw_value: str) -> Any:
    value = raw_value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip("\"'")


def _parse_gin_assignment(expression: str) -> Optional[tuple[str, Any]]:
    if "=" not in expression:
        return None
    key, raw_value = expression.split("=", 1)
    key = key.strip()
    if "." not in key:
        return None
    field_name = key.rsplit(".", 1)[1]
    if field_name not in _CONFIG_FIELDS:
        return None
    return field_name, _parse_literal(raw_value)


def _load_gin_values_with_fallback(
    gin_configs: Iterable[str], gin_bindings: Iterable[str]
) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for gin_config in gin_configs:
        with open(gin_config, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parsed = _parse_gin_assignment(stripped)
                if parsed is not None:
                    key, value = parsed
                    values[key] = value

    for binding in gin_bindings:
        parsed = _parse_gin_assignment(binding)
        if parsed is not None:
            key, value = parsed
            values[key] = value
    return values


def _load_gin_values(gin_configs: Iterable[str], gin_bindings: Iterable[str]) -> Dict[str, Any]:
    if gin is None:
        return _load_gin_values_with_fallback(gin_configs, gin_bindings)

    gin.clear_config()
    try:
        for gin_config in gin_configs:
            gin.parse_config_file(gin_config)
        if gin_bindings:
            gin.parse_config(list(gin_bindings))
        config = SingleDayLaunchConfig()
        return {field: getattr(config, field) for field in _CONFIG_FIELDS}
    finally:
        gin.clear_config()


def build_config_from_sources(
    gin_config: Optional[str] = None,
    gin_configs: Optional[Iterable[str]] = None,
    gin_bindings: Iterable[str] = (),
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> SingleDayLaunchConfig:
    config_paths = list(gin_configs) if gin_configs is not None else []
    if gin_config:
        config_paths = [gin_config] + config_paths

    values = _load_gin_values(config_paths, gin_bindings)
    config = SingleDayLaunchConfig(**values)
    for key, value in (cli_overrides or {}).items():
        if key in _CONFIG_FIELDS:
            setattr(config, key, value)
    return config


def apply_launch_config(args, config, explicit):
    for arg_key, config_key in _ARG_TO_CONFIG_KEY.items():
        if config_key in explicit:
            continue
        if hasattr(args, arg_key):
            setattr(args, arg_key, getattr(config, config_key))
    return args


def extract_explicit_config_keys(argv: Iterable[str]) -> Set[str]:
    explicit: Set[str] = set()
    for token in argv:
        if token.startswith("--"):
            normalized = token.split("=", 1)[0]
            config_key = _CLI_OPTION_TO_CONFIG_KEY.get(normalized)
            if config_key is not None:
                explicit.add(config_key)
    return explicit
