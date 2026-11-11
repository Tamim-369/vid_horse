"""Machine profile and resource caps.

POLICY: strictly one worker everywhere. The pipeline once over-paralleled
(2 Chatterbox TTS lines, 3-wide LLM race waves, multi-slot ollama) and a
mid-size box could hard-crash when ollama models, TTS, whisper and ffmpeg
held memory at the same time. Worker counts are 1 now — wall clock suffers
a little; the machine survives always.

The other half of the guarantee lives in the ollama SERVER env (set once
in the service):

    OLLAMA_NUM_PARALLEL=1        # one request slot, no parallel KV caches
    OLLAMA_MAX_LOADED_MODELS=1   # never keep two models resident at once
    OLLAMA_KEEP_ALIVE=1m         # unload promptly between stages
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


def _total_gb() -> float:
    try:
        text = Path("/proc/meminfo").read_text()
        for line in text.splitlines():
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError, IndexError):
        pass
    return 8.0


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
        total_gb = _total_gb()
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


# -- worker presets (ALL 1 — see module docstring) ----------------------------


def voice_workers() -> int:
    """TTS lines render strictly one at a time: two Chatterbox copies at
    once was the single biggest RAM spike in the pipeline."""

    return 1


def torch_threads() -> int:
    """Intra-op threads for small torch models (whisper, CLIP)."""

    return 1


def voice_torch_threads() -> int:
    """Chatterbox gets half the cores (system keeps breathing room)."""

    return max(2, machine().cores // 2)


def set_torch_threads() -> None:
    """Pin intra-op threads to the agreed burst size (imported lazily)."""

    try:
        import torch

        torch.set_num_threads(torch_threads())
        torch.set_num_interop_threads(1)
    except Exception:  # pragma: no cover - torch absent in some envs
        logger.info("torch not importable; thread pin skipped")


# -- thermal guard ------------------------------------------------------------
# This box hard-shutdowns at ~88C (kernel thermal trip) and idles hot. An
# all-core CPU burst (local LLM inference, TTS, ffmpeg) can cross the line
# in under a minute, so the batch checks before it starts.

def cpu_temp_c() -> float | None:
    """Package CPU temperature in degrees C, or None if unreadable."""

    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if "k10temp" in (hwmon / "name").read_text().lower() or "coretemp" in (
                hwmon / "name"
            ).read_text().lower():
                for tf in sorted(hwmon.glob("temp*_input")):
                    return int(tf.read_text().strip()) / 1000.0
        except (OSError, ValueError):
            continue
    for tf in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            return int(tf.read_text().strip()) / 1000.0
        except (OSError, ValueError):
            continue
    return None


def thermal_status(soft_c: float = 70.0, hard_c: float = 80.0) -> str:
    """``"ok"`` / ``"warm"`` / ``"critical"`` from the live CPU temp."""

    temp = cpu_temp_c()
    if temp is None:
        return "ok"
    if temp >= hard_c:
        return "critical"
    if temp >= soft_c:
        return "warm"
    return "ok"


def wait_for_cool(soft_c: float = 70.0, timeout_s: float = 300.0) -> bool:
    """Block until the CPU cools below ``soft_c`` (or the timeout passes)."""

    import time as _time

    deadline = _time.monotonic() + timeout_s
    while _time.monotonic() < deadline:
        if thermal_status(soft_c=soft_c, hard_c=soft_c) == "ok":
            return True
        _time.sleep(10.0)
    return thermal_status(soft_c=soft_c, hard_c=soft_c) == "ok"
