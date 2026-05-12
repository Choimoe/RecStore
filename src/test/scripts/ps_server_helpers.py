#!/usr/bin/env python3

import os
import socket
import json
import glob


RDMA_SKIP_EXIT_CODE = 77


def find_ps_server_binary():
    """Find ps_server binary in common locations."""
    server_path = os.environ.get('PS_SERVER_PATH')
    if server_path:
        return os.path.abspath(server_path)
    
    candidates = [
        './bin/ps_server',
        './build/bin/ps_server',
        '../bin/ps_server',
        '../../build/bin/ps_server',
        '../../../build/bin/ps_server',
        '../../../../build/bin/ps_server',
    ]
    
    for candidate in candidates:
        abs_candidate = os.path.abspath(candidate)
        if os.path.exists(abs_candidate):
            return abs_candidate
    
    return os.path.abspath('./build/bin/ps_server')


def is_port_open(host, port, timeout=1):
    """Check if a port is open/listening."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def find_config_file():
    """Find recstore_config.json by searching upwards to the root."""
    config_path = os.environ.get('RECSTORE_CONFIG')
    if config_path and os.path.exists(config_path):
        return os.path.abspath(config_path)
    
    current_dir = os.path.abspath(os.getcwd())
    while True:
        candidate = os.path.join(current_dir, 'recstore_config.json')
        if os.path.exists(candidate):
            return candidate
        
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached root
            break
        current_dir = parent_dir
    
    return None


def load_config():
    """Load the active RecStore config file."""
    config_path = find_config_file()
    if not config_path:
        return None, {}

    with open(config_path, 'r') as f:
        return config_path, json.load(f)


def get_backend_type():
    """Return the configured backend type for the current test run."""
    _config_path, config = load_config()
    cache_ps = config.get('cache_ps', {})
    return str(cache_ps.get('ps_type', 'GRPC')).upper()


def get_rdma_runner_config():
    """Extract the RDMA runner settings needed by the PetPS test harness."""
    _config_path, config = load_config()
    cache_ps = config.get('cache_ps', {})
    dist_client = config.get('distributed_client', {})
    base_kv = cache_ps.get('base_kv_config', {})
    return {
        'num_servers': int(dist_client.get('num_shards', cache_ps.get('num_shards', 1))),
        'value_size': int(base_kv.get('value', {}).get('default_value_size_hint', base_kv.get('value_size', 512))),
        'max_kv_num_per_request': int(dist_client.get('max_keys_per_request', 64)),
    }


def get_rdma_skip_reason():
    """Return skip reason when RDMA verbs devices are not available."""
    rdma_device_dir = '/dev/infiniband'
    if not os.path.isdir(rdma_device_dir):
        return f"RDMA verbs device directory is unavailable: {rdma_device_dir}"

    uverbs_devices = sorted(glob.glob(os.path.join(rdma_device_dir, 'uverbs*')))
    if not uverbs_devices:
        return f"RDMA verbs devices are unavailable under {rdma_device_dir}"

    return None


def get_ports_from_config():
    """Extract ports from recstore_config.json."""
    config_path, config = load_config()
    if not config_path:
        return [15000, 15001, 15002, 15003]

    try:
        ports = []
        # Try to get ports from cache_ps.servers
        cache_ps = config.get('cache_ps', {})
        servers = cache_ps.get('servers', [])
        for s in servers:
            if 'port' in s:
                ports.append(s['port'])
        
        # If no ports found in cache_ps, try distributed_client.servers
        if not ports:
            dist_client = config.get('distributed_client', {})
            servers = dist_client.get('servers', [])
            for s in servers:
                if 'port' in s:
                    ports.append(s['port'])
        
        # Fallback if still no ports
        if not ports:
            return [15000, 15001, 15002, 15003]
            
        return sorted(list(set(ports)))
    except Exception:
        return [15000, 15001, 15002, 15003]


def check_ps_server_running(ports=None):
    """Check if ps_server is running by checking if ports are open."""
    if ports is None:
        ports = get_ports_from_config()
    
    open_ports = [port for port in ports if is_port_open('127.0.0.1', port)]
    
    if open_ports:
        return True, open_ports
    return False, []


def should_skip_server_start():
    """Determine if we should skip starting ps_server."""
    is_ci = os.environ.get('CI') == 'true' or os.environ.get('GITHUB_ACTIONS') == 'true'
    no_server = os.environ.get('NO_PS_SERVER', '').lower() in ('1', 'true', 'yes')
    configured_ports = get_ports_from_config()
    running, open_ports = check_ps_server_running(configured_ports)
    all_ports_ready = bool(configured_ports) and len(open_ports) == len(configured_ports)
    partial_ports_open = running and not all_ports_ready
    
    if no_server:
        return True, "NO_PS_SERVER"

    if partial_ports_open:
        raise RuntimeError(
            f"ps_server ports are partially available: expected={configured_ports}, open={open_ports}"
        )

    if is_ci:
        if all_ports_ready:
            return True, f"ci_reuse_running:{open_ports}"
        return False, f"ci_server_not_ready: expected={configured_ports}, open={open_ports}"
    
    if all_ports_ready:
        return True, f"already_running:{open_ports}"
    
    return False, None


def get_server_config():
    """Get server configuration from environment."""
    return {
        'server_path': find_ps_server_binary(),
        'config_path': os.environ.get('RECSTORE_CONFIG'),
        'log_dir': os.environ.get('PS_LOG_DIR', './logs'),
        'timeout': int(os.environ.get('PS_TIMEOUT', '60')),
        'num_shards': int(os.environ.get('PS_NUM_SHARDS', '2')),
    }
