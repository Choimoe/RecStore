#!/usr/bin/env python3

import os
import queue
import re
import shutil
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _to_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class PetPSClusterRunner:
    def __init__(
        self,
        server_path="./build/bin/petps_server",
        config_path="./recstore_config.json",
        num_servers=1,
        num_clients=1,
        thread_num=1,
        value_size=16,
        max_kv_num_per_request=64,
        timeout=60,
        startup_delay=2.0,
        log_dir="/tmp",
        verbose=False,
        memcached_host="127.0.0.1",
        memcached_port=21211,
        use_local_memcached="auto",
        memcached_check_timeout=2.0,
        memcached_check_retries=3,
        status_refresh_interval=2.0,
        show_status_logs=True,
        show_memcached_logs=True,
        memcached_namespace="auto",
        rdma_per_thread_response_limit_bytes=None,
        rdma_server_ready_timeout_sec=None,
        rdma_server_ready_poll_ms=None,
        rdma_client_receive_arena_bytes=None,
        rdma_put_protocol_version=None,
        rdma_put_v2_transfer_mode=None,
        rdma_put_v2_push_slot_bytes=None,
        rdma_put_v2_push_slots_per_client=None,
        rdma_put_v2_push_region_offset=None,
        rdma_put_client_send_arena_bytes=None,
        rdma_put_server_scratch_bytes=None,
        rdma_wait_timeout_ms=None,
        rdma_transport_mode=None,
        rdma_transport_mode_client_flag=True,
        rdma_use_dram=True,
        validate_routing=False,
    ):
        self.server_path = Path(server_path)
        if not self.server_path.is_absolute():
            self.server_path = (REPO_ROOT / self.server_path).resolve()

        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = (REPO_ROOT / self.config_path).resolve()
        self.num_servers = num_servers
        self.num_clients = num_clients
        self.thread_num = thread_num
        self.value_size = value_size
        self.max_kv_num_per_request = max_kv_num_per_request
        self.timeout = timeout
        self.startup_delay = startup_delay
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        self.memcached_host = memcached_host
        self.memcached_port = memcached_port
        self.use_local_memcached = use_local_memcached
        self.memcached_check_timeout = memcached_check_timeout
        self.memcached_check_retries = memcached_check_retries
        self.status_refresh_interval = status_refresh_interval
        self.show_status_logs = show_status_logs
        self.show_memcached_logs = show_memcached_logs
        if memcached_namespace == "auto":
            memcached_namespace = f"recstore-{os.getpid()}-{time.time_ns()}"
        self.memcached_namespace = memcached_namespace
        self.rdma_per_thread_response_limit_bytes = (
            rdma_per_thread_response_limit_bytes
        )
        self.rdma_server_ready_timeout_sec = rdma_server_ready_timeout_sec
        self.rdma_server_ready_poll_ms = rdma_server_ready_poll_ms
        self.rdma_client_receive_arena_bytes = rdma_client_receive_arena_bytes
        self.rdma_put_protocol_version = rdma_put_protocol_version
        self.rdma_put_v2_transfer_mode = rdma_put_v2_transfer_mode
        self.rdma_put_v2_push_slot_bytes = rdma_put_v2_push_slot_bytes
        self.rdma_put_v2_push_slots_per_client = rdma_put_v2_push_slots_per_client
        self.rdma_put_v2_push_region_offset = rdma_put_v2_push_region_offset
        self.rdma_put_client_send_arena_bytes = rdma_put_client_send_arena_bytes
        self.rdma_put_server_scratch_bytes = rdma_put_server_scratch_bytes
        self.rdma_wait_timeout_ms = rdma_wait_timeout_ms
        self.rdma_transport_mode = rdma_transport_mode
        self.rdma_transport_mode_client_flag = rdma_transport_mode_client_flag
        self.rdma_use_dram = rdma_use_dram
        self.validate_routing = validate_routing
        self.processes = []
        self.process_logs = {}
        self.memcached_process = None
        self.ready = set()
        self.ready_threads = {}

    def _allocate_local_memcached_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.memcached_host, 0))
            return sock.getsockname()[1]

    def emit_status(self, phase, extra=""):
        if not self.show_status_logs:
            return
        running_pids = [
            str(process.pid)
            for process, _thread in self.processes
            if process.poll() is None
        ]
        detail = (
            f" phase={phase} ready={len(self.ready)}/{self.num_servers}"
            f" running_pids={','.join(running_pids) if running_pids else 'none'}"
        )
        if extra:
            detail += f" {extra}"
        print(f"[petps-status]{detail}")

    def build_env(self):
        env = os.environ.copy()
        env["RECSTORE_MEMCACHED_HOST"] = self.memcached_host
        env["RECSTORE_MEMCACHED_PORT"] = str(self.memcached_port)
        env["RECSTORE_MEMCACHED_TEXT_PROTOCOL"] = "1"
        if self.memcached_namespace:
            env["RECSTORE_MEMCACHED_NAMESPACE"] = self.memcached_namespace
        if self.rdma_transport_mode is not None:
            env["RECSTORE_RDMA_TRANSPORT_MODE"] = self.rdma_transport_mode
        if self.validate_routing:
            env["RECSTORE_RDMA_VALIDATE_ROUTING"] = "1"
        return env

    def build_memcached_cmd(self):
        memcached_bin = shutil.which("memcached")
        if memcached_bin is None:
            raise RuntimeError(
                "memcached binary not found in PATH; install memcached or use "
                "--use-local-memcached=never with an external memcached "
                "instance"
            )
        cmd = [memcached_bin]
        # Only pass -u root when actually running as root.
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            cmd.extend(["-u", "root"])
        cmd.extend([
            "-l",
            self.memcached_host,
            "-p",
            str(self.memcached_port),
            "-c",
            "10000",
        ])
        return cmd

    def build_server_cmd(self, global_id):
        cmd = [
            str(self.server_path),
            f"--config_path={self.config_path}",
            f"--global_id={global_id}",
            f"--num_server_processes={self.num_servers}",
            f"--num_client_processes={self.num_clients}",
            f"--thread_num={self.thread_num}",
            f"--value_size={self.value_size}",
            f"--max_kv_num_per_request={self.max_kv_num_per_request}",
            f"--use_dram={'true' if self.rdma_use_dram else 'false'}",
        ]
        if self.rdma_per_thread_response_limit_bytes is not None:
            cmd.append(
                "--rdma_per_thread_response_limit_bytes="
                f"{self.rdma_per_thread_response_limit_bytes}"
            )
        if self.rdma_put_server_scratch_bytes is not None:
            cmd.append(
                "--rdma_put_server_scratch_bytes="
                f"{self.rdma_put_server_scratch_bytes}"
            )
        if self.rdma_put_v2_push_slot_bytes is not None:
            cmd.append(
                "--rdma_put_v2_push_slot_bytes="
                f"{self.rdma_put_v2_push_slot_bytes}"
            )
        if self.rdma_put_v2_push_slots_per_client is not None:
            cmd.append(
                "--rdma_put_v2_push_slots_per_client="
                f"{self.rdma_put_v2_push_slots_per_client}"
            )
        if self.rdma_put_v2_push_region_offset is not None:
            cmd.append(
                "--rdma_put_v2_push_region_offset="
                f"{self.rdma_put_v2_push_region_offset}"
            )
        if self.rdma_transport_mode is not None:
            cmd.append(f"--rdma_transport_mode={self.rdma_transport_mode}")
        return cmd

    def build_client_cmd(self, argv, client_index=0):
        client_global_id = self.num_servers + client_index
        cmd = list(argv) + [
            f"--global_id={client_global_id}",
            f"--num_server_processes={self.num_servers}",
            f"--num_client_processes={self.num_clients}",
            f"--value_size={self.value_size}",
            f"--max_kv_num_per_request={self.max_kv_num_per_request}",
        ]
        if self.rdma_server_ready_timeout_sec is not None:
            cmd.append(
                "--rdma_server_ready_timeout_sec="
                f"{self.rdma_server_ready_timeout_sec}"
            )
        if self.rdma_server_ready_poll_ms is not None:
            cmd.append(f"--rdma_server_ready_poll_ms={self.rdma_server_ready_poll_ms}")
        if self.rdma_client_receive_arena_bytes is not None:
            cmd.append(
                "--rdma_client_receive_arena_bytes="
                f"{self.rdma_client_receive_arena_bytes}"
            )
        if self.rdma_put_protocol_version is not None:
            cmd.append(
                "--rdma_put_protocol_version="
                f"{self.rdma_put_protocol_version}"
            )
        if self.rdma_put_v2_transfer_mode is not None:
            cmd.append(
                "--rdma_put_v2_transfer_mode="
                f"{self.rdma_put_v2_transfer_mode}"
            )
        if self.rdma_put_v2_push_slot_bytes is not None:
            cmd.append(
                "--rdma_put_v2_push_slot_bytes="
                f"{self.rdma_put_v2_push_slot_bytes}"
            )
        if self.rdma_put_v2_push_slots_per_client is not None:
            cmd.append(
                "--rdma_put_v2_push_slots_per_client="
                f"{self.rdma_put_v2_push_slots_per_client}"
            )
        if self.rdma_put_v2_push_region_offset is not None:
            cmd.append(
                "--rdma_put_v2_push_region_offset="
                f"{self.rdma_put_v2_push_region_offset}"
            )
        if self.rdma_put_client_send_arena_bytes is not None:
            cmd.append(
                "--rdma_put_client_send_arena_bytes="
                f"{self.rdma_put_client_send_arena_bytes}"
            )
        if self.rdma_wait_timeout_ms is not None:
            cmd.append(f"--rdma_wait_timeout_ms={self.rdma_wait_timeout_ms}")
        if (
            self.rdma_transport_mode is not None
            and self.rdma_transport_mode_client_flag
        ):
            cmd.append(f"--rdma_transport_mode={self.rdma_transport_mode}")
        return cmd

    def is_ready_line(self, line):
        return (
            "[RDMA-DBG] Server polling thread ready" in line
            or "component=rdma_server event=polling_thread_ready" in line
        )

    def _extract_ready_thread_token(self, line):
        if "component=rdma_server event=polling_thread_ready" in line:
            match = re.search(r"thread_id=(\d+)", line)
            if match is not None:
                return match.group(1)
        return line.rsplit(" ", 1)[-1]

    def _monitor(self, global_id, pipe):
        for raw_line in iter(pipe.readline, ""):
            line = raw_line.rstrip()
            self.process_logs.setdefault(global_id, []).append(line)
            if self.verbose:
                print(f"[petps_server:{global_id}] {line}")
            if self.is_ready_line(line):
                ready = self.ready_threads.setdefault(global_id, set())
                ready.add(self._extract_ready_thread_token(line))
                if len(ready) >= self.thread_num:
                    self.ready.add(global_id)

    def _format_captured_process_output(self, global_id):
        lines = self.process_logs.get(global_id, [])
        if not lines:
            return ""
        return (
            f"\nCaptured output from petps_server[{global_id}] "
            f"(last {len(lines)} lines):\n" + "\n".join(lines)
        )

    def _start_memcached(self):
        cmd = self.build_memcached_cmd()
        self.memcached_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
        )
        if self.show_memcached_logs:
            print(
                f"[petps-memcached] started with pid={self.memcached_process.pid} "
                f"host={self.memcached_host} port={self.memcached_port}"
            )

        time.sleep(0.2)
        if self.memcached_process.poll() is not None:
            log = ""
            if self.memcached_process.stdout is not None:
                log = self.memcached_process.stdout.read()
            err = (
                f"memcached exited early with code {self.memcached_process.returncode}"
            )
            if log:
                err += f"\nmemcached output:\n{log}"
            raise RuntimeError(err)

    def check_memcached_ready(self):
        last_error = None
        for attempt in range(1, self.memcached_check_retries + 1):
            try:
                with socket.create_connection(
                    (self.memcached_host, self.memcached_port),
                    timeout=self.memcached_check_timeout,
                ) as sock:
                    sock.settimeout(self.memcached_check_timeout)
                    sock.sendall(b"get serverNum\r\n")
                    data = sock.recv(4096)
                    if b"END\r\n" in data or b"VALUE serverNum" in data:
                        self.emit_status(
                            "memcached-ready",
                            f"attempt={attempt} host={self.memcached_host}:{self.memcached_port}",
                        )
                        return
                    last_error = RuntimeError(
                        "memcached responded but without expected get reply"
                    )
            except OSError as exc:
                last_error = exc
            self.emit_status(
                "memcached-wait",
                f"attempt={attempt}/{self.memcached_check_retries} "
                f"host={self.memcached_host}:{self.memcached_port}",
            )
            time.sleep(0.2)

        raise RuntimeError(
            "memcached is not reachable or not ready at "
            f"{self.memcached_host}:{self.memcached_port}; "
            "set --use-local-memcached=always|auto|never and "
            "RECSTORE_MEMCACHED_HOST/RECSTORE_MEMCACHED_PORT as needed"
        ) from last_error

    def reset_memcached_state(self):
        def key(name):
            if self.memcached_namespace:
                return f"{self.memcached_namespace}:{name}"
            return name

        server_num_key = key("serverNum")
        client_num_key = key("clientNum")
        dsm_key = key("xmh-consistent-dsm")

        payload_lines = []
        # Namespace mode isolates keys per run, so no global flush_all is needed.
        if not self.memcached_namespace:
            payload_lines.append("flush_all")
        payload_lines.extend(
            [
                f"set {server_num_key} 0 0 1",
                "0",
                f"set {client_num_key} 0 0 1",
                "0",
                f"set {dsm_key} 0 0 1",
                "1",
                f"get {server_num_key}",
                f"get {client_num_key}",
                f"get {dsm_key}",
                "quit",
            ]
        )
        payload = ("\r\n".join(payload_lines) + "\r\n").encode("ascii")
        with socket.create_connection(
            (self.memcached_host, self.memcached_port),
            timeout=self.memcached_check_timeout,
        ) as sock:
            sock.settimeout(self.memcached_check_timeout)
            sock.sendall(payload)
            response = b""
            while True:
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    break
                if not chunk:
                    break
                response += chunk
            if (
                (not self.memcached_namespace and b"OK\r\n" not in response)
                or response.count(b"STORED\r\n") < 3
                or f"VALUE {server_num_key} 0 1\r\n0\r\n".encode("ascii")
                not in response
                or f"VALUE {client_num_key} 0 1\r\n0\r\n".encode("ascii")
                not in response
                or f"VALUE {dsm_key} 0 1\r\n1\r\n".encode("ascii")
                not in response
            ):
                raise RuntimeError(
                    "failed to initialize memcached state; "
                    f"response was: {response!r}"
                )
            self.emit_status(
                "memcached-reset",
                f"host={self.memcached_host}:{self.memcached_port}",
            )

    def _prepare_memcached(self):
        if self.use_local_memcached not in {"always", "auto", "never"}:
            raise ValueError(
                "use_local_memcached must be one of: always, auto, never"
            )

        if self.use_local_memcached == "always":
            self._start_memcached()
            self.check_memcached_ready()
            self.reset_memcached_state()
            return

        try:
            self.check_memcached_ready()
            self.reset_memcached_state()
            return
        except RuntimeError:
            if self.use_local_memcached != "auto":
                raise

        # Existing memcached may be unreachable or incompatible with the reset
        # sequence we require. In auto mode, prefer starting local memcached on
        # the requested endpoint (typically 127.0.0.1:21211) so legacy code
        # paths that still read memcached.conf continue to work.
        self.memcached_host = "127.0.0.1"
        requested_port = self.memcached_port
        try:
            self._start_memcached()
        except RuntimeError:
            self.memcached_port = self._allocate_local_memcached_port()
            self.emit_status(
                "memcached-auto-port-fallback",
                (
                    f"requested={self.memcached_host}:{requested_port} "
                    f"fallback={self.memcached_host}:{self.memcached_port}"
                ),
            )
            self._start_memcached()
        self.check_memcached_ready()
        self.reset_memcached_state()

    def start(self):
        if not self.server_path.exists():
            raise FileNotFoundError(f"Server binary not found: {self.server_path}")

        self._prepare_memcached()
        env = self.build_env()
        if self.show_memcached_logs:
            print(
                "[petps-memcached] server env "
                f"host={env['RECSTORE_MEMCACHED_HOST']} "
                f"port={env['RECSTORE_MEMCACHED_PORT']}"
            )

        for global_id in range(self.num_servers):
            process = subprocess.Popen(
                self.build_server_cmd(global_id),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(REPO_ROOT),
                env=env,
            )
            thread = threading.Thread(
                target=self._monitor, args=(global_id, process.stdout), daemon=True
            )
            thread.start()
            self.processes.append((process, thread))

        if self.startup_delay > 0:
            time.sleep(self.startup_delay)

        # PetPS servers cannot reach polling-ready until the client joins the
        # DSM-init barrier. If no post-barrier ready line appears yet, let the
        # client proceed as long as the server process is still alive; the RDMA
        # client waits on a memcached server-ready key before sending RPCs.
        if not self.ready:
            for global_id, (process, _thread) in enumerate(self.processes):
                if process.poll() is None:
                    self.ready.add(global_id)

        deadline = time.time() + self.timeout
        next_refresh = time.time() + self.status_refresh_interval
        while len(self.ready) < self.num_servers:
            if time.time() > deadline:
                self.emit_status("startup-timeout", f"timeout={self.timeout}s")
                self.stop()
                raise TimeoutError(
                    f"Timed out waiting for {self.num_servers} petps_server processes"
                )
            for idx, (process, _thread) in enumerate(self.processes):
                if process.poll() is not None:
                    self.emit_status(
                        "startup-crash",
                        f"exit_code={process.returncode}",
                    )
                    crash_details = self._format_captured_process_output(idx)
                    self.stop()
                    raise RuntimeError(
                        "petps_server exited early with code "
                        f"{process.returncode}{crash_details}"
                    )
            if (
                self.status_refresh_interval > 0
                and time.time() >= next_refresh
            ):
                self.emit_status("startup-wait")
                next_refresh = time.time() + self.status_refresh_interval
            time.sleep(0.2)

    def run_client(self, argv, client_index=0, stream_output=True, timeout=None):
        cmd = self.build_client_cmd(argv, client_index=client_index)
        env = self.build_env()
        if self.show_memcached_logs:
            print(
                "[petps-memcached] client env "
                f"host={env['RECSTORE_MEMCACHED_HOST']} "
                f"port={env['RECSTORE_MEMCACHED_PORT']} "
                f"client_index={client_index}"
            )
        if not stream_output:
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=str(REPO_ROOT),
                    text=True,
                    capture_output=True,
                    check=False,
                    env=env,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                class Completed:
                    def __init__(self, stdout, stderr):
                        self.returncode = 124
                        self.stdout = stdout
                        self.stderr = stderr

                timeout_text = (
                    f"\n[petps-client] timed out after {timeout} seconds\n"
                )
                stdout = _to_text(exc.stdout) + timeout_text
                stderr = _to_text(exc.stderr)
                return Completed(stdout, stderr)
            if self.verbose:
                print(completed.stdout)
                print(completed.stderr)
            return completed

        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=env,
        )

        output_lines = []
        line_queue = queue.Queue()

        def read_stdout():
            try:
                for line in iter(process.stdout.readline, ""):
                    if not line:
                        break
                    line_queue.put(line)
            finally:
                line_queue.put(None)

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()

        deadline = time.monotonic() + timeout if timeout is not None else None
        returncode = None
        timed_out = False
        reader_done = False

        while True:
            try:
                line = line_queue.get(timeout=0.05)
                if line is None:
                    reader_done = True
                else:
                    output_lines.append(line)
                    print(line, end="")
            except queue.Empty:
                pass

            if returncode is None:
                returncode = process.poll()

            if (
                deadline is not None
                and time.monotonic() >= deadline
                and returncode is None
            ):
                timed_out = True
                break

            if returncode is not None and reader_done:
                break

        if timed_out:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            timeout_line = f"[petps-client] timed out after {timeout} seconds\n"
            output_lines.append(timeout_line)
            print(timeout_line, end="")
            returncode = 124
        elif returncode is None:
            returncode = process.wait()

        reader.join(timeout=1)
        while True:
            try:
                line = line_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                continue
            output_lines.append(line)
            print(line, end="")

        class Completed:
            def __init__(self, returncode, stdout):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = ""

        return Completed(returncode, "".join(output_lines))

    def stop(self):
        for process, thread in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            thread.join(timeout=1)
        self.processes.clear()
        self.process_logs.clear()
        if self.memcached_process is not None and self.memcached_process.poll() is None:
            self.memcached_process.terminate()
            try:
                self.memcached_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.memcached_process.kill()
                self.memcached_process.wait(timeout=5)
        self.memcached_process = None

    @contextmanager
    def run(self):
        self.start()
        try:
            yield self
        finally:
            self.stop()
