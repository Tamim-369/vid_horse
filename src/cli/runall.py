"""Budgeted slug scheduler: many videos per night from one machine.

Runs whole slugs (script → assets → voice → render) back-to-back across
a queue while keeping the box honest:

- admission is measured, not guessed — a stage-job starts only when its
  profile-derived RAM/CPU weight fits inside what the machine reports as
  available *right now*, and a disk lane is free (HDD boxes get exactly
  one lane so parallel encodes never seek-thrash the platter);
- every job is a plain `videoai` subprocess, so each one fails, logs and
  resumes exactly like a manual run — nothing about the output or any
  quality gate changes with concurrency;
- a watchdog samples process trees (/proc): logs utilization, kills a
  runaway before swap, and re-queues the stage (which resumes from its
  own caches — wav cache, media digests, LLM cache);
- finished stages are detected through their on-disk markers, so a re-run
  of the same queue skips what is already done and continues where the
  last run froze.

Weak boxes are the design target: profiles set every worker count, so
the same code maxes a 12-core/14GB workstation and stays inside an old
i3's 6 usable GB and spinning disk.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from utils import computing
from utils.computing import machine

logger = logging.getLogger(__name__)
console = Console()

# swap must never happen; stage caches make a respawn nearly free.
# The hard rule: 20% of total RAM and 20% of the cores stay untouched by
# the scheduler, always — admission, the starvation valve and the watchdog
# all measure against that reserve, never past it.
_RAM_RESERVE_FRACTION = 0.20
_CPU_RESERVE_FRACTION = 0.20


def _reserve_gb() -> float:
    """RAM the scheduler must leave free (20% of total, min 1G)."""

    return max(1.0, machine().total_gb * _RAM_RESERVE_FRACTION)


def _cpu_busy_fraction(sample: float) -> float:
    """Share of all cores busy between two /proc/stat samples."""

    try:
        fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        idle_a = float(fields[3]) + float(fields[4] if len(fields) > 4 else 0)
        total_a = sum(float(v) for v in fields)
    except (OSError, ValueError, IndexError):
        return 0.0
    prev = _CPU_SAMPLE.get("data")
    _CPU_SAMPLE["data"] = (idle_a, total_a)
    if not prev:
        return 0.0
    idle_b, total_b = prev
    dt = total_a - total_b
    if dt <= 0:
        return 0.0
    return 1.0 - (idle_a - idle_b) / dt


_CPU_SAMPLE: dict[str, tuple[float, float]] = {}
_STAGE_TIMEOUT = {
    "script": 90 * 60,
    "assets": 4 * 60 * 60,
    "voice": 2 * 60 * 60,
    "render": 90 * 60,
}


@dataclass(frozen=True)
class Stage:
    """All the scheduler knows about one unit of work."""

    name: str
    args: list[str]
    marker: Path


def _meminfo_available_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value = line.split(":", 1)
            if name.strip() == "MemAvailable":
                return float(value.split()[0]) / 1024 / 1024
    except (OSError, ValueError):
        pass
    return 2.0


def _tree(pid: int) -> list[int]:
    """PIDs of the process and all descendants (one /proc walk)."""

    seen = {pid}
    frontier = [pid]
    parents: dict[int, int] = {}
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(") ", 1)[-1].split()
                parents[int(entry.name)] = int(fields[1])
            except (OSError, ValueError, IndexError):
                continue
    except OSError:
        return list(seen)
    while frontier:
        head = frontier.pop()
        hit = [proc for proc, parent in parents.items() if parent == head]
        frontier.extend(p for p in hit if p not in seen)
        seen.update(hit)
    return list(seen)


def _tree_rss_gb(pid: int) -> float:
    total = 0.0
    for proc in _tree(pid):
        try:
            for line in Path(f"/proc/{proc}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += float(line.split()[1]) / 1024 / 1024
                    break
        except (OSError, ValueError):
            continue
    return total


def _external_model_gb() -> float:
    """RSS of ollama's llama-server, if resident.

    The script stage's memory lives mostly in the ollama server, not in the
    CLI subprocess the scheduler spawns — count it explicitly while scripts
    are in flight so the 20% reserve stays honest.
    """

    total = 0.0
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ")
                if b"llama-server" not in cmdline:
                    continue
                for line in (entry / "status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        total += float(line.split()[1]) / 1024 / 1024
                        break
            except (OSError, ValueError):
                continue
    except OSError:
        pass
    return total


def _tree_cpu_seconds(pid: int) -> float:
    total = 0.0
    for proc in _tree(pid):
        try:
            fields = Path(f"/proc/{proc}/stat").read_text().rsplit(") ", 1)[-1].split()
            total += (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
        except (OSError, ValueError, IndexError):
            continue
    return total


@dataclass
class Live:
    """A stage-job in flight. ``covers`` lists ever slug it produces
    outputs for (batch voice jobs cover several slugs at once)."""

    slug: str
    stage: Stage
    ram_weight: float = 0.0
    cpu_weight: float = 0.0
    proc: subprocess.Popen = None  # type: ignore[assignment]
    log: Path = None  # type: ignore[assignment]
    covers: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)
    cpu_prev: float = 0.0
    tick_prev: float = 0.0
    cpu_sum: float = 0.0
    cpu_n: int = 0
    rss_peak: float = 0.0


@dataclass
class Track:
    """One slug's progress through its chain."""

    slug: str
    fmt: str
    stages: list[Stage]
    restarts: dict = field(default_factory=dict)  # stage name -> count
    pos: int = 0
    failed: str | None = None

    @property
    def done(self) -> bool:
        return self.pos >= len(self.stages)


def build_tracks(
    slugs: list[str], fmt: str, output_dir: Path, force: bool
) -> list[Track]:
    """Chain per slug; stages already on disk are skipped outright."""

    stem = "voice" if fmt == "short" else f"voice-{fmt}"
    tracks: list[Track] = []
    for slug in slugs:
        directory: Path | None = output_dir / slug
        markers = [
            Stage("script", ["script", slug, "--format", fmt],
                  directory / f"{fmt}.json"),
            Stage("assets", ["assets", slug, "--format", fmt],
                  directory / f"assets-{fmt}.json"),
            Stage("voice", ["voice", slug, "--format", fmt],
                  directory / "audio" / f"{stem}.wav"),
            Stage("render", ["render", slug, "--format", fmt],
                  directory / f"{slug}-{fmt}.mp4"),
        ]
        if not force:
            markers = [stage for stage in markers if not stage.marker.exists()]
        tracks.append(Track(slug=slug, fmt=fmt, stages=markers))
    return tracks


def run_queue(
    *,
    slugs: list[str],
    fmt: str = "short",
    ram_budget: float | None = None,
    force: bool = False,
    output_dir: Path | None = None,
    log_root: Path | None = None,
) -> dict[str, str]:
    """Run every slug through its remaining stages; return a status map."""

    mode = machine()
    base = output_dir
    if base is None:
        configured = os.environ.get("VIDEOAI_OUTPUT_DIR")
        base = Path(configured) if configured else Path("output")
    log_dir = (log_root or base / "scheduler") / time.strftime("%Y%m%d-%H%M%S")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        log_dir = Path("/tmp") / log_dir.name

    # Reserve 20% of total RAM to keep system stable
    default_budget = max(2.0, mode.total_gb * (1.0 - _RAM_RESERVE_FRACTION))
    budget = min(
        ram_budget if ram_budget is not None else default_budget,
        max(2.0, mode.total_gb * (1.0 - _RAM_RESERVE_FRACTION)),
    )
    lanes = computing.disk_lanes()
    tracks = build_tracks(slugs, fmt, base, force)
    pending = [track for track in tracks if track.stages]
    for track in tracks:
        if not track.stages:
            console.print(f"[yellow]{track.slug}: nothing to do[/yellow]")
    console.print(
        f"[bold]machine:[/bold] {mode.cores} cores / {mode.total_gb:.1f}G "
        f"({'HDD' if mode.hdd else 'SSD'}), budget {budget:.1f}G, "
        f"{lanes} disk lane(s), {len(pending)} slug(s) to run"
    )

    live: list[Live] = []
    statuses: dict[str, str] = {}
    total_cpu_sum, total_cpu_n = 0.0, 0
    idle_ticks = 0

    while pending or live:
        if os.environ.get("VIDEOAI_SCHED_DEBUG"):
            live_slugs = [(e.slug, e.stage.name) for e in live]
            tick_dbg = [
                (t.slug, t.pos, t.stages[t.pos].name, t.failed)
                for t in pending
            ]
            console.print(f"[dim]tick: live={live_slugs} pending={tick_dbg}[/dim]")
        now = time.monotonic()

        # -- reap finished jobs
        for entry in list(live):
            proc = entry.proc
            code = proc.poll()
            if code is None:
                continue
            minutes = (now - entry.started) / 60
            stage_key = f"{entry.slug}:{entry.stage.name}"
            done_well = code == 0
            statuses[stage_key] = "ok" if done_well else "failed"
            if done_well:
                console.print(
                    f"[green]✓ {entry.stage.name} ({entry.slug}"
                    + (f"+{len(entry.covers) - 1} more" if len(entry.covers) > 1 else "")
                    + f") {minutes:.1f}m, peak {entry.rss_peak:.1f}G, "
                    f"avg {entry.cpu_sum / max(entry.cpu_n, 1):.1f} cores[/green]"
                )
                for slug_covered in entry.covers or [entry.slug]:
                    track = next(t for t in tracks if t.slug == slug_covered)
                    track.pos += 1
                    if track.done:
                        statuses[slug_covered] = "ok"
                        console.print(
                            f"[bold green]{slug_covered}: complete[/bold green]"
                        )
            else:
                track = next(t for t in tracks if t.slug == entry.slug)
                count = track.restarts.get(entry.stage.name, 0) + 1
                track.restarts[entry.stage.name] = count
                statuses[f"{entry.slug}:{entry.stage.name}"] = "failed"
                if count <= 2:
                    # a failed subprocess often just hit a 429 storm or a
                    # full-disk transient; its own caches (LLM entries,
                    # finished line wavs, media digests) make the respawn
                    # cheap, so overnight batches plow through instead
                    # of dying on one unlucky slug
                    console.print(
                        f"[yellow]⟲ {entry.stage.name} ({entry.slug}"
                        + (
                            f"+{len(entry.covers) - 1} more"
                            if len(entry.covers) > 1
                            else ""
                        )
                        + f" ) failed (rc={code}); retry {count}/2 queued[/yellow]"
                    )
                    live.remove(entry)
                    continue
                console.print(
                    f"[red]✗ {entry.stage.name} ({entry.slug}) FAILED "
                    f"rc={code} after {count} start(s) — {entry.log}[/red]"
                )
                for slug_covered in entry.covers or [entry.slug]:
                    cover_track = next(t for t in tracks if t.slug == slug_covered)
                    cover_track.failed = f"{entry.stage.name}: rc {code}"
                    statuses[slug_covered] = "failed"
            live.remove(entry)

        # -- watchdog: utilization + timeouts + swap protection
        available = _meminfo_available_gb()
        util_samples: list[float] = []
        for entry in live:
            rss = _tree_rss_gb(entry.proc.pid)
            entry.rss_peak = max(entry.rss_peak, rss)
            cpu_now = _tree_cpu_seconds(entry.proc.pid)
            if entry.cpu_prev:
                used = (cpu_now - entry.cpu_prev) / max(now - entry.tick_prev, 1e-6)
                entry.cpu_sum += used
                entry.cpu_n += 1
                util_samples.append(used)
            entry.cpu_prev, entry.tick_prev = cpu_now, now
            if now - entry.started > _STAGE_TIMEOUT.get(entry.stage.name, 60 * 60):
                console.print(
                    f"[red]{entry.slug} {entry.stage.name}: timed out[/red]"
                )
                _stop(entry.proc)
                live.remove(entry)
                track = next(t for t in tracks if t.slug == entry.slug)
                track.failed = f"{entry.stage.name}: timeout"
                statuses[entry.slug] = "failed"
                statuses[f"{entry.slug}:{entry.stage.name}"] = "failed"
            elif available < _reserve_gb() and live and entry.rss_peak >= max(
                e.rss_peak for e in live
            ):
                # the box is about to swap: release the biggest resident
                console.print(
                    f"[red]RAM floor: releasing {entry.slug} "
                    f"{entry.stage.name} ({entry.rss_peak:.1f}G); resuming "
                    "later from its own caches[/red]"
                )
                _stop(entry.proc)
                live.remove(entry)
                track = next(t for t in tracks if t.slug == entry.slug)
                ram_key = f"{entry.stage.name}#ram"
                track.restarts[ram_key] = track.restarts.get(ram_key, 0) + 1
                if track.restarts[ram_key] >= 6:
                    track.failed = f"{entry.stage.name}: kept hitting the RAM floor"
                    statuses[entry.slug] = "failed"
        if util_samples:
            total_cpu_sum += sum(util_samples) / len(util_samples)
            total_cpu_n += 1
        pending = [t for t in tracks if not t.done and not t.failed]

        # -- admission: per slug at most one job; per stage the weights
        ram_used = sum(entry.ram_weight for entry in live)
        script_busy = any(entry.stage.name == "script" for entry in live) or any(
            t.stages and t.pos < len(t.stages) and t.stages[t.pos].name == "script"
            for t in pending
        )
        if script_busy:
            ram_used += _external_model_gb()
        cpu_used = sum(entry.cpu_weight for entry in live)
        disk_used = sum(
            1
            for entry in live
            if computing.stage_weight(entry.stage.name).disk >= 0.5
        )
        covered = {slug for entry in live for slug in (entry.covers or [entry.slug])}
        reserve = _reserve_gb()
        admit_now = min(budget, _meminfo_available_gb() - reserve)
        cpu_cap = min(
            mode.cores_budget, max(1.0, mode.cores * (1.0 - _CPU_RESERVE_FRACTION))
        )
        busy = _cpu_busy_fraction(now)
        if busy > 1.0 - _CPU_RESERVE_FRACTION:
            # the box (or ollama's llama-server) is already chewing >80% of
            # the cores: add nothing this tick, just breathe
            if os.environ.get("VIDEOAI_SCHED_DEBUG"):
                console.print(f"[dim]cpu busy {busy:.0%}: holding admission[/dim]")
            admit_now = float("-inf")

        def _spawn(label: str, args: list[str], covers: list[str]):
            dry = os.environ.get("VIDEOAI_SCHEDULER_DRY") == "1"
            command = ["sleep", "2"] if dry else ["uv", "run", "videoai", *args]
            log_path = log_dir / f"{label}.log"
            try:
                with open(log_path, "ab") as handle:
                    proc = subprocess.Popen(
                        command,
                        cwd=_repo_root(),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
            except OSError as exc:
                console.print(f"[red]cannot spawn {label}: {exc}[/red]")
                for covered_slug in covers:
                    t = next(t for t in tracks if t.slug == covered_slug)
                    t.failed = f"{label}: spawn failed"
                    statuses[covered_slug] = "failed"
                return None
            return proc

        # -- voice batching: every pending slug whose next stage is voice
        # goes into ONE process, so the Chatterbox model loads once for the
        # whole batch instead of once per slug
        voice_ready = [
            t
            for t in pending
            if t.slug not in covered and t.stages and t.stages[t.pos].name == "voice"
        ]
        others = [
            t
            for t in pending
            if t.slug not in covered and t.stages and t.stages[t.pos].name != "voice"
        ]
        if not live and (voice_ready or others) and idle_ticks >= 30:
            # starvation valve: the box is quiet and every next stage still
            # fails admission (~1 min straight) — stop deferring to whatever
            # other apps eat MemAvailable. The boost stays anchored to real
            # memory pressure: only healthy-pressure boxes (>= 3G free)
            # get the +1.5G nudge; below that the scheduler waits instead
            # of gambling the machine into swap. The RAM-floor watchdog
            # still releases residents before things touch swap.
            available = _meminfo_available_gb()
            room = available - reserve
            needed = (
                4.2
                if voice_ready
                else min(
                    (computing.stage_weight(t.stages[t.pos].name).ram_gb
                     for t in others),
                    default=0.0,
                )
            )
            if room >= needed:
                admit_now = max(admit_now, min(budget, room))
                console.print(
                    "[yellow]starvation valve: boosting admission "
                    f"({available:.1f}G free, {reserve:.1f}G reserved)[/yellow]"
                )
            idle_ticks = 0
        batch = []
        if voice_ready:
            # size the batch to the measured room, not a fixed cap: one
            # model load + 1.2G per extra slug, never past the reserve
            room = admit_now - ram_used
            n = min(len(voice_ready), 4)
            while n > 1 and 4.2 + 1.2 * (n - 1) > room:
                n -= 1
            if n >= 1 and room >= 4.2:
                batch = voice_ready[:n]
        if batch:
            slugs = [t.slug for t in batch]
            args = ["voice", *slugs, "--format", fmt]
            ram = 4.2 + 1.2 * (len(slugs) - 1)
            cpu = 3.0 + 0.5 * (len(slugs) - 1)
            marker = batch[0].stages[batch[0].pos]
            stage = Stage("voice", args, marker)
            if os.environ.get("VIDEOAI_SCHED_DEBUG"):
                console.print(
                    f"[dim]voice-batch: ready={slugs} ram={ram:.1f}+{ram_used:.1f}"
                    f"<={admit_now:.1f} cpu={cpu:.1f}+{cpu_used:.1f}"
                    f"<={mode.cores_budget:.1f} disk={disk_used}<{lanes}[/dim]"
                )
            if (
                ram_used + ram <= admit_now
                and cpu_used + cpu <= cpu_cap
                and disk_used < lanes
            ):
                proc = _spawn("voice-batch-" + "-".join(slugs), args, slugs)
                if proc is not None:
                    live.append(
                        Live(
                            slug=slugs[0],
                            stage=stage,
                            ram_weight=ram,
                            cpu_weight=cpu,
                            covers=slugs,
                            proc=proc,
                            log=log_dir / f"voice-{'-'.join(slugs)}.log",
                            tick_prev=time.monotonic(),
                        )
                    )
                    ram_used += ram
                    cpu_used += cpu
                    label = "voice" if len(slugs) == 1 else f"voice batch x{len(slugs)}"
                    console.print(f"[cyan]▶ {label}[/cyan] (weight {ram:.1f}G / {cpu:.1f}c)")

        for track in others:
            if track.slug in covered:
                continue
            stage = track.stages[track.pos]
            weight = computing.stage_weight(stage.name)
            if (
                ram_used + weight.ram_gb > admit_now
                or cpu_used + weight.cpu > cpu_cap
                or (weight.disk >= 0.5 and disk_used >= lanes)
            ):
                continue
            proc = _spawn(f"{track.slug}-{stage.name}", stage.args, [track.slug])
            if proc is None:
                continue
            live.append(
                Live(
                    slug=track.slug,
                    stage=stage,
                    ram_weight=weight.ram_gb,
                    cpu_weight=weight.cpu,
                    proc=proc,
                    log=log_dir / f"{track.slug}-{stage.name}.log",
                    tick_prev=time.monotonic(),
                )
            )
            ram_used += weight.ram_gb
            cpu_used += weight.cpu
            disk_used += 1 if weight.disk >= 0.5 else 0
            console.print(
                f"[cyan]▶ {track.slug} {stage.name}[/cyan] "
                f"(weight {weight.ram_gb:.1f}G / {weight.cpu:.0f}c)"
            )

        time.sleep(2.0)
        if live:
            idle_ticks = 0
        else:
            idle_ticks += 1

    done_slugs = sum(
        1
        for track in tracks
        if statuses.get(track.slug) == "ok" or (not track.stages and not track.failed)
    )
    failed = sum(1 for value in statuses.values() if value == "failed")
    if total_cpu_n:
        console.print(
            f"[bold]{len(tracks)} slug(s): avg {(total_cpu_sum / total_cpu_n):.1f} "
            f"cores of {cpu_cap:.0f} in budget; {done_slugs} done, "
            f"{failed} failed[/bold]"
        )
    return statuses


def _stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), 15)
        proc.wait(timeout=20)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), 9)


def _repo_root() -> str:
    # src/cli/runall.py → src → repo root (where .cache/ and pyproject live)
    return str(Path(__file__).resolve().parents[2])
