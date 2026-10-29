"""Machine profile: what this box can afford to run at once.

Every concurrency knob in the pipeline derives from here, so the same
codebase saturates a 12-core workstation and stays inside the swap line
of an 8GB i3 with a spinning disk. Detection is pure /proc + sysfs —
no third-party dependency, works on Linux containers and cgroups.

The hero number is ``available_gb``: the budget the scheduler may hand
out to running stage jobs. Stage calls themselves scale their worker
counts by ``cores`` and honour ``set_torch_threads`` for their exact
burst, never assuming they own the machine.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Machine:
    """The one machine: counts and limits, measured at import time."""

    cores: int
    total_gb: float
    hdd: bool
    tiny: bool

    @property
    def cores_budget(self) -> float:
        # Reserve 20% of CPU cores to keep system responsive
        # measurements say a full render takes ~7 cores yet tolerates a
        # voice/asset neighbour without slowing; 80% ensures OS breathing room
        return max(1.0, self.cores * 0.8)

    def available_gb(self) -> float:
        """Free RAM right now (fresh read), minus a safe kernel margin."""

        for path in ("/proc/meminfo",):
            try:
                info: dict[str, float] = {}
                for line in Path(path).read_text().splitlines():
                    name, value = line.split(":", 1)
                    info[name.strip()] = float(value.split()[0])
                mem_available = info.get("MemAvailable", info.get("MemTotal", 0.0))
                # Reserve 20% of resources to keep system stable
                return max(0.5, mem_available * 0.80 / 1024 / 1024)
            except (OSError, ValueError):
                logger.info("could not read %s; assuming a humble box", path)
                return 2.0
        return 2.0


@dataclass(frozen=True)
class StageWeight:
    """What one stage-job costs while it holds a scheduler slot."""

    ram_gb: float
    cpu: float
    disk: float  # 0 = chatty network reads, 1 = multi-stream encode IO


def _cgroup_cores() -> int | None:
    try:
        raw = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if raw == ["max", "max"]:
            return None
        return max(1, math.floor(int(raw[0]) / int(raw[1])))
    except (OSError, IndexError, ValueError):
        return None


def _disk_is_rotational(base: Path) -> bool:
    try:
        inode = os.stat(base)
        major, minor = os.major(inode.st_dev), os.minor(inode.st_dev)
        entry = Path(f"/sys/dev/block/{major}:{minor}")
        for candidate in (entry, *entry.parents):
            queue = candidate / "queue" / "rotational"
            if queue.exists():
                return queue.read_text().strip() == "1"
            slaves = candidate / "slaves"
            if slaves.is_dir():
                names = [n for n in slaves.iterdir() if n.name != "loop*"]
                if names:
                    probe = Path("/sys/dev/block") / names[0].name / "queue" / "rotational"
                    return probe.exists() and probe.read_text().strip() == "1"
    except OSError:
        return False
    return False


def _meminfo() -> tuple[float, float]:
    try:
        text = Path("/proc/meminfo").read_text()
        totals: dict[str, float] = {}
        for line in text.splitlines():
            name, value = line.split(":", 1)
            totals[name.strip()] = float(value.split()[0])
        total = totals.get("MemTotal", 0.0) / 1024 / 1024
        available_raw = totals.get("MemAvailable", totals.get("MemTotal", 0.0))
        # Reserve 20% of resources to keep system stable
        available = available_raw * 0.80 / 1024 / 1024
        return total, max(0.5, available)
    except (OSError, ValueError):
        return 8.0, 2.0


_machine: Machine | None = None


def machine() -> Machine:
    """Detect once, reuse everywhere; safe to call from any thread."""

    global _machine
    if _machine is None:
        try:
            cores = len(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover - non-Linux fallback
            cores = os.cpu_count() or 2
        cg = _cgroup_cores()
        if cg:
            cores = min(cores, cg)
        total_gb, _ = _meminfo()
        root = Path(os.environ.get("VIDEOAI_DATA_DIR", ".")).resolve()
        hdd = _disk_is_rotational(root)
        tiny = cores <= 4 or total_gb <= 6.0
        _machine = Machine(cores=cores, total_gb=total_gb, hdd=hdd, tiny=tiny)
        logger.info(
            "machine profile: %s cores, %.1fG ram, %s, %s",
            _machine.cores,
            _machine.total_gb,
            "HDD" if _machine.hdd else "SSD",
            "tiny" if _machine.tiny else "full",
        )
    return _machine


# -- per-stage worker presets ------------------------------------------------
def scene_workers() -> int:
    return 2 if machine().tiny else 3


def clip_workers() -> int:
    # per-scene pool; total torch inference threads = scenes × this
    return 2 if machine().tiny else 3


def torch_threads() -> int:
    return 1 if machine().tiny else 2


def voice_torch_threads() -> int:
    """Voice stage runs mostly alone: give each model its fair share."""

    return max(1, machine().cores // max(1, voice_workers()))


def variant_workers() -> int:
    return 2 if machine().tiny else 3


def voice_workers() -> int:
    return 1 if machine().tiny else 2


def render_workers() -> int:
    """
    Measured on the 12-core box: 3 concurrent full-default x264 encodes
    hit 715% CPU and 70s wall; adding workers or capping per-encode
    threads slowed the wall clock. Keep the proven shape.
    """

    return 2 if machine().tiny else 3


def ffmpeg_threads() -> int:
    """
    x264 auto-threads win wall clock (see render_workers()): the caps
    stay OFF for real encodes. Kept for console-side probes only.
    """

    return 0


def race_attempts() -> int:
    """LLM corrective wave size once the serial retry has burned once."""

    return 2 if machine().tiny else 3


def disk_lanes() -> int:
    """Concurrent disk-heavy stage jobs (renders, still encodes)."""

    return 1 if machine().hdd else 2


def stage_weight(stage: str) -> StageWeight:
    tiny = machine().tiny
    if tiny:
        return {
            "research": StageWeight(ram_gb=0.5, cpu=1.0, disk=0.0),
            "script": StageWeight(ram_gb=0.5, cpu=1.0, disk=0.0),
            "assets": StageWeight(ram_gb=1.2, cpu=3.0, disk=0.2),
            "voice": StageWeight(ram_gb=3.2, cpu=3.0, disk=0.1),
            "render": StageWeight(ram_gb=3.4, cpu=2.5, disk=1.0),
        }[stage]
    return {
        "research": StageWeight(ram_gb=0.8, cpu=1.0, disk=0.0),
        "script": StageWeight(ram_gb=0.8, cpu=1.0, disk=0.0),
        "assets": StageWeight(ram_gb=1.8, cpu=3.0, disk=0.2),
        "voice": StageWeight(ram_gb=4.2, cpu=3.0, disk=0.1),
        "render": StageWeight(ram_gb=4.5, cpu=5.0, disk=1.0),
    }[stage]


def set_torch_threads() -> None:
    """Pin intra-op threads to the agreed burst size (imported lazily)."""

    try:
        import torch

        torch.set_num_threads(torch_threads())
        torch.set_num_interop_threads(1)
    except Exception:  # pragma: no cover - torch absent in some envs
        logger.info("torch not importable; thread pin skipped")
