"""Out-of-process parameter search that can only ever produce a PROPOSAL.

This package exists because of one invariant in PROJECT_MAP.md §11: the AI is
advisory. Nothing here may change the active strategy, write settings, or place
an order — the best a search can do is rank parameter sets and (via
:mod:`algotrading.optimize.proposal`) leave a PENDING `AIRecommendation` for a
human to approve or reject in Telegram.

Why a separate process (`runner` + `spawn`): the bot's 60-second tick, the
Telegram handlers and APScheduler all share one interpreter. A grid search is
CPU-bound for minutes, so running it in-process would starve tick handling, delay
protective exits and look exactly like a hang to the supervisor's heartbeat — and
on Android a thread that pins a core is what the phantom-process killer targets.
The child is spawned with `nice` and a wall-clock timeout, and a lockfile makes
sure two searches never run at once.

Modules:

  * `search`   — grid generation, fold aggregation and candidate ranking.
  * `runner`   — the CLI the child process runs (`python -m algotrading.optimize.runner`).
  * `spawn`    — start/wait on that child with nice, timeout and a lockfile.
  * `proposal` — turn a finished search into a PENDING recommendation.
"""
from __future__ import annotations

__all__ = ["search", "runner", "spawn", "proposal"]
