"""Runtime facts the container knows but Python does not ask about.

The important one is the CPU count. `os.cpu_count()` reports the *host's*
cores and ignores the cgroup quota the container actually runs under. On a
Railway 8 vCPU service sitting on a 64-core host it returns 64, so any library
that sizes its thread pool that way spawns 64 workers for 8 cores. That costs
twice: context-switch contention, and one memory arena per thread - which is
how a ~3.5GB workload grew to 7GB and started hitting the 8GB ceiling.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager


def cpu_limit() -> int:
    """Cores this process may actually use, honouring cgroup limits."""
    override = os.environ.get("CPU_LIMIT")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass

    # cgroup v2 - "<quota> <period>", or "max <period>" when unrestricted
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, round(int(quota) / int(period)))
    except Exception:
        pass

    # cgroup v1
    try:
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if quota > 0:
            return max(1, round(quota / period))
    except Exception:
        pass

    # respects taskset/affinity even when no cgroup quota is set
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 2)


@contextmanager
def stage(name: str):
    """Time a pipeline stage into the container logs.

    Railway shows stdout, so a slow deploy can be diagnosed from the log
    instead of by reading CPU graphs.
    """
    t0 = time.time()
    print(f"[stage] {name} ...", flush=True)
    try:
        yield
    finally:
        print(f"[stage] {name} done in {time.time() - t0:.1f}s", flush=True)


def log_memory(note: str = "") -> None:
    """Current and peak RSS, for spotting a run creeping toward the ceiling.

    Both numbers matter: peak never falls, so only the current figure shows
    whether freeing a model actually returned the memory.
    """
    current = peak = None
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                current = int(line.split()[1]) / 1024
            elif line.startswith("VmHWM:"):
                peak = int(line.split()[1]) / 1024
    except Exception:
        pass

    if peak is None:
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return

    now = f"{current:.0f} MB" if current is not None else "?"
    print(f"[mem] now {now}, peak {peak:.0f} MB {note}".rstrip(), flush=True)
