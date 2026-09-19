"""Start the parameter search as a SEPARATE OS process, niced and time-boxed.

WHY IT MUST NEVER RUN INSIDE THE SCHEDULER THREAD
    The bot runs one interpreter. The 60-second market tick, the Telegram
    handlers and APScheduler's worker thread all share it, and CPython's GIL
    means a CPU-bound grid search running in a thread still steals the tick's
    time slices. Concretely, running the search in-process would:

      * delay the tick that closes a position or fires a stop (money, not
        politeness);
      * make the heartbeat/supervisor believe the bot is hung (a log that stops
        advancing mid-tick is exactly the signature they watch for);
      * put an OOM or a crash in the search on the bot's own process;
      * and on Android, a thread that pins a CPU core is precisely what the
        phantom-process killer targets, so the whole app gets SIGKILLed.

    So the search always runs in a child process (`python -m
    algotrading.optimize.runner`), started at `optimize.nice` so the tick keeps
    its CPU, and killed at `optimize.timeout_seconds`. A lockfile makes sure two
    searches never compete for the same phone.

The child can read candles and write a JSON artifact. It cannot approve
anything: approval lives in the Telegram flow, and `proposal` only ever leaves a
PENDING row.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

LOCK_NAME = "search.lock"
DEFAULT_TIMEOUT_SECONDS = 900
DEFAULT_NICE = 10
RUNNER_MODULE = "algotrading.optimize.runner"


class SearchBusyError(RuntimeError):
    """Another search holds the lockfile (or its pid is still alive)."""


@dataclass
class SearchHandle:
    """A running child: the process, where it writes, and the lock it holds."""

    proc: subprocess.Popen
    out_path: Path
    log_path: Path
    lock_path: Path
    command: list[str] = field(default_factory=list)
    log_handle: Any = None

    @property
    def pid(self) -> int | None:
        return self.proc.pid


def results_dir(settings: Any) -> Path:
    """The artifact directory, created on demand (Termux: never CWD-relative)."""
    path = Path(getattr(getattr(settings, "optimize", None), "results_dir", "data/optimize"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def lock_path(settings: Any) -> Path:
    return results_dir(settings) / LOCK_NAME


def _pid_alive(pid: int) -> bool:
    """True when a process with this pid exists (signal 0 probes without killing)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    except OSError:  # pragma: no cover - platform specific
        return False
    return True


def acquire_lock(settings: Any) -> Path:
    """Take the single-search lock or raise `SearchBusyError`.

    A lock whose pid is gone (killed search, reboot, SIGKILL from the Android
    killer) is STALE and gets replaced — otherwise one dead child would disable
    the feature until someone deleted a file by hand. A live holder always wins,
    including this same process: "I already started one search" is precisely the
    case the lock exists for, and `wait_search` always releases it.
    """
    path = lock_path(settings)
    payload = {"pid": os.getpid(), "started_at": time.time()}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8") or "{}")
        except (ValueError, OSError):
            existing = {}
        holder = int(existing.get("pid", 0) or 0)
        if holder and _pid_alive(holder):
            raise SearchBusyError(
                f"a parameter search is already running (pid {holder}); "
                f"remove {path} if that process is gone"
            )
        log.warning("optimize: replacing stale search lock %s (pid %s)", path, holder)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def release_lock(path: Path) -> None:
    """Drop the lock, ignoring the case where it is already gone."""
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:  # pragma: no cover - permissions
        log.warning("optimize: could not remove lock %s: %s", path, exc)


def build_command(
    *,
    strategy: str,
    symbol: str | None = None,
    interval: str | None = None,
    combinations: int | None = None,
    out_path: Path | None = None,
    python: str | None = None,
    nice: int | None = None,
    settings: Any = None,
) -> list[str]:
    """The child command line (`nice` first when the platform has it)."""
    if nice is None:
        nice = int(getattr(getattr(settings, "optimize", None), "nice", DEFAULT_NICE) or 0)
    interpreter = (
        python
        or getattr(getattr(settings, "optimize", None), "trainer_python", "")
        or sys.executable
    )
    command: list[str] = []
    # `nice` as an external prefix rather than preexec_fn=os.nice: preexec_fn is
    # unsafe in a threaded parent (the bot is one), `nice(1)` is not.
    if nice and shutil.which("nice"):
        command += ["nice", "-n", str(int(nice))]
    command += [interpreter, "-m", RUNNER_MODULE, "--strategy", strategy]
    if symbol:
        command += ["--symbol", symbol]
    if interval:
        command += ["--interval", interval]
    if combinations:
        command += ["--combinations", str(int(combinations))]
    if out_path is not None:
        command += ["--out", str(out_path)]
    return command


def start_search_subprocess(
    *,
    strategy: str,
    symbol: str | None = None,
    interval: str | None = None,
    combinations: int | None = None,
    out: str | Path | None = None,
    timeout: int | None = None,
    settings: Any = None,
    python: str | None = None,
    nice: int | None = None,
) -> SearchHandle:
    """Launch the search as a separate, niced, time-boxed OS process.

    Raises:
        SearchBusyError: another search holds the lock.
        RuntimeError: the child could not be started (lock released again).
    """
    if settings is None:
        from algotrading.config import load_settings

        settings = load_settings()
    directory = results_dir(settings)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = strategy.replace("/", "_").replace(" ", "_")
    out_path = Path(out) if out else directory / f"search_{safe}_{stamp}.json"
    if not out_path.is_absolute():
        # A relative --out still belongs to the artifact dir, not to whatever
        # directory the supervisor happened to start the bot from.
        out_path = directory / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log")

    lock = acquire_lock(settings)
    command = build_command(
        strategy=strategy, symbol=symbol, interval=interval,
        combinations=combinations, out_path=out_path,
        python=python, nice=nice, settings=settings,
    )
    try:
        from algotrading.config import PROJECT_ROOT

        cwd = str(PROJECT_ROOT)
    except Exception:  # pragma: no cover - no config tree
        cwd = os.getcwd()
    try:
        log_handle = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as exc:
        release_lock(lock)
        raise RuntimeError(f"could not start the search process: {exc}") from exc

    handle = SearchHandle(
        proc=proc,
        out_path=out_path,
        log_path=log_path,
        lock_path=lock,
        command=command,
        log_handle=log_handle,
    )
    log.info(
        "optimize: started search pid=%s nice=%s timeout=%ss -> %s",
        proc.pid, nice, timeout, out_path,
    )
    return handle


def wait_search(
    handle: SearchHandle,
    *,
    timeout: int | None = None,
    settings: Any = None,
) -> int:
    """Wait for the child, kill it at `timeout`, and always release the lock.

    Returns the child's exit code, or -9 when it had to be killed. The lock is
    released in a `finally` so a crashed or timed-out search never leaves the
    feature permanently busy.
    """
    if timeout is None:
        timeout = int(
            getattr(getattr(settings, "optimize", None), "timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
            or DEFAULT_TIMEOUT_SECONDS
        )
    code = -9
    try:
        try:
            code = handle.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.warning(
                "optimize: search pid %s exceeded %ss; killing it", handle.pid, timeout
            )
            handle.proc.kill()
            try:
                handle.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - unkillable child
                log.error("optimize: search pid %s did not die", handle.pid)
            code = -9
    finally:
        if handle.log_handle is not None:
            try:
                handle.log_handle.close()
            except OSError:  # pragma: no cover
                pass
        release_lock(handle.lock_path)
    return code


__all__ = [
    "LOCK_NAME",
    "SearchBusyError",
    "SearchHandle",
    "acquire_lock",
    "build_command",
    "lock_path",
    "release_lock",
    "results_dir",
    "start_search_subprocess",
    "wait_search",
]
