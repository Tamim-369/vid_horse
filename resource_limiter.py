#!/usr/bin/env python3
"""
Resource limiter wrapper - enforces CPU/memory limits on subprocesses.
Usage: python resource_limiter.py --cpu 50 --mem 6144 -- python3 -c "code here"
"""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    print("Installing psutil...")
    subprocess.run([sys.executable, "-m", "pip", "install", "psutil"], check=True)
    import psutil


class ResourceLimiter:
    def __init__(self, cpu_percent: float, memory_mb: float, check_interval: float = 1.0):
        self.cpu_limit = cpu_percent / 100.0 * psutil.cpu_count()
        self.memory_limit = memory_mb * 1024 * 1024
        self.check_interval = check_interval
        self.process = None
        self._stop_monitoring = False
        self.monitor_thread = None

    def start(self, cmd: list[str], cwd: Path | None = None) -> int:
        self.process = subprocess.Popen(cmd, cwd=cwd, start_new_session=True)
        self._stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()
        try:
            return self.process.wait()
        finally:
            self._stop_monitoring = True
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2)

    def _monitor(self):
        while not self._stop_monitoring and self.process.poll() is None:
            try:
                self._check_resources()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.check_interval)

    def _check_resources(self):
        if not self.process or self.process.poll() is not None:
            return
        try:
            p = psutil.Process(self.process.pid)
            children = p.children(recursive=True)
            all_procs = [p] + children
            total_mem = sum(proc.memory_info().rss for proc in all_procs)
            if total_mem > self.memory_limit:
                print(f"⚠️  Memory limit exceeded: {total_mem / 1024 / 1024:.0f}MB > {self.memory_limit / 1024 / 1024:.0f}MB")
                self._throttle_processes(all_procs)
            total_cpu = sum(proc.cpu_percent(interval=0.1) for proc in all_procs)
            if total_cpu > self.cpu_limit * 100:
                print(f"⚠️  CPU limit exceeded: {total_cpu:.1f}% > {self.cpu_limit * 100:.1f}%")
                self._throttle_processes(all_procs)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def _throttle_processes(self, procs: list):
        for proc in procs:
            try:
                proc.nice(10)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass


def main():
    args = sys.argv[1:]
    cpu = 50
    mem = 8192
    interval = 1.0
    cwd = None
    i = 0
    while i < len(args):
        if args[i] == "--":
            i += 1
            break
        elif args[i] == "--cpu":
            cpu = float(args[i + 1])
            i += 2
            continue
        elif args[i] == "--mem":
            mem = float(args[i + 1])
            i += 2
            continue
        elif args[i] == "--interval":
            interval = float(args[i + 1])
            i += 2
            continue
        elif args[i] == "--cwd":
            cwd = Path(args[i + 1])
            i += 2
            continue
        else:
            break
        i += 1
    command = args[i:]
    if not command:
        print("Error: No command provided. Use -- to separate options from command.")
        sys.exit(1)
    limiter = ResourceLimiter(cpu_percent=cpu, memory_mb=mem, check_interval=1.0)
    cwd = cwd or Path.cwd()
    print(f"🔒 Starting with limits: CPU={cpu}%, Mem={mem}MB")
    print(f"📁 CWD: {cwd}")
    print(f"🚀 Command: {' '.join(command)}")
    try:
        exit_code = limiter.start(command, cwd=cwd)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()