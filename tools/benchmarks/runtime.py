from __future__ import annotations

import json
from pathlib import Path

from .common import DEFAULT_OUTPUT_ROOT, REPO_ROOT


def build_remote_recstore_runtime(run_slug: str, *, path_name: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    base_cfg = json.loads((REPO_ROOT / 'recstore_config.json').read_text(encoding='utf-8'))
    runtime_dir = output_root / 'runtime' / run_slug
    runtime_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(base_cfg))
    cfg['cache_ps']['ps_type'] = 'BRPC'
    cfg['cache_ps']['num_shards'] = 2
    cfg['cache_ps']['servers'] = [
        {'host': '10.0.2.68', 'port': 15123, 'shard': 0},
        {'host': '10.0.2.68', 'port': 15124, 'shard': 1},
    ]
    cfg['distributed_client'] = {
        'num_shards': 2,
        'hash_method': 'city_hash',
        'max_keys_per_request': 500,
        'servers': [
            {'host': '10.0.2.68', 'port': 15123, 'shard': 0},
            {'host': '10.0.2.68', 'port': 15124, 'shard': 1},
        ],
    }
    cfg['client'] = {'host': '10.0.2.68', 'port': 15123, 'shard': 0}
    base_kv = cfg['cache_ps']['base_kv_config']
    base_kv['capacity'] = 10400000
    base_kv['index'] = {'type': 'DRAM_EXTENDIBLE_HASH'}
    value = base_kv.setdefault('value', {})
    value['type'] = 'DRAM_VALUE_STORE'
    value['path'] = f'/tmp/{path_name}/value'
    value['dram_allocator'] = {
        'type': 'PERSIST_LOOP_SLAB',
        'capacity_bytes': 10400000 * int(value.get('default_value_size_hint', 512)),
    }
    config_path = runtime_dir / 'recstore_config.json'
    config_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    return config_path
