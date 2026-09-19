"""Turn a finished search into a PENDING recommendation. Nothing more.

The hard rule (PROJECT_MAP.md §11, "AI advisory only"): whatever proposes a
change, the change waits for a human. So this module is deliberately small and
deliberately powerless:

  * it writes ONE row per proposal, always with `status='pending'`;
  * it never imports `RecommendationStore.apply`, `create_new_version`,
    `promote_to_active` or anything from `algotrading.store.strategy_versions` —
    there is no code path here that can activate a strategy;
  * it never writes `config/settings.yaml` or any settings object.

Why it does not go through `create_pending_recommendation`: that function is the
validated entry point for a proposal (kind allow-list, parameter dict, strategy
name resolution) but it has no parameter for the search evidence, and
`AIRecommendation.backtest_json` is exactly where the evidence belongs — the
approve/reject card and the decision log read it. So the row is created through
the public API and the evidence is attached to that same session immediately
after, which keeps the validation gate without bypassing it.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from sqlalchemy.orm import Session

from algotrading.db.models import AIRecommendation
from algotrading.optimize.search import Candidate, SearchResult
from algotrading.store.recommendations import PENDING, create_pending_recommendation

log = logging.getLogger(__name__)

KIND = "param_change"


def _evidence(result: SearchResult, top_n: int) -> dict[str, Any]:
    """The search facts a human needs to judge the proposal, and no more."""
    best = result.best()
    return {
        "source": "optimize.search",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strategy_name": result.strategy_name,
        "symbol": result.symbol,
        "interval": result.interval,
        "objective": result.objective,
        "n_bars": result.n_bars,
        "folds": result.folds,
        "min_trades": result.min_trades,
        "n_candidates": result.n_candidates,
        "n_evaluated": result.n_evaluated,
        "n_rejected": result.n_rejected,
        "seed": result.seed,
        "duration_seconds": round(result.duration_seconds, 3),
        "timed_out": result.timed_out,
        "best": best.to_dict(rank=1) if best is not None else None,
        "top": [c.to_dict(rank=i + 1) for i, c in enumerate(result.ranked[:top_n])],
        "rejected_sample": [c.to_dict() for c in result.rejected[:5]],
        "ranked": len(result.ranked),
        # The invariants a reviewer is entitled to see, next to the numbers.
        "invariants": [
            "This is a PROPOSAL from an in-sample walk-forward search; it is not "
            "applied, and it stays pending until a human approves it.",
            "No execution, settings or active-strategy code was touched by the search.",
        ],
    }


def _rationale(
    result: SearchResult, best: Candidate, *, rationale: str | None = None
) -> str:
    if rationale:
        return rationale
    consistency = best.consistency or {}
    value = best.objective_value
    shown = "n/a" if value is None else f"{value:.4f}"
    folds = consistency.get("folds", result.folds)
    profitable = consistency.get("profitable_folds")
    metrics = best.metrics or {}
    return (
        f"Parameter search ({result.objective}) over {result.n_candidates} candidate(s) "
        f"on {result.symbol} {result.interval} ({result.n_bars} bars, {folds} sequential "
        f"walk-forward folds): best {result.objective}={shown} with "
        f"{metrics.get('n_trades')} trades, {profitable}/{folds} folds profitable, "
        f"worst fold drawdown {metrics.get('worst_fold_drawdown_pct')}%. "
        f"Proposed params: {best.params}. "
        f"{consistency.get('verdict', '')} Pending human approval."
    )


def propose_from_result(
    session: Session,
    result: SearchResult,
    *,
    rationale: str | None = None,
    top_n: int = 3,
    position_pct: float | None = None,
) -> AIRecommendation:
    """Create a PENDING `param_change` recommendation from a search result.

    Args:
        session: an open DB session (the row is committed before returning).
        result: a finished `SearchResult` (see `runner` / `scripts/optimize.py`).
        rationale: override the generated text.
        top_n: how many ranked candidates to embed as evidence.
        position_pct: optional sizing hint for the recommendation content.

    Returns:
        The committed `AIRecommendation` with `status == 'pending'`.

    Raises:
        ValueError: the search produced nothing scoreable (proposing an empty
            search would create an unreviewable card), or the strategy name is
            not buildable — the same gate the assistant's proposals go through.
    """
    best = result.best()
    if best is None:
        raise ValueError(
            f"refusing to propose from a search with no ranked candidate "
            f"({result.n_rejected} rejected / {result.n_candidates} candidates)"
        )
    if not best.params:
        raise ValueError("refusing to propose an empty parameter set")

    # The public, validated path: kind allow-list, dict check, strategy name
    # resolution. It always leaves status='pending' and never applies anything.
    rec = create_pending_recommendation(
        session,
        kind=KIND,
        strategy_name=result.strategy_name,
        params=dict(best.params),
        rationale=_rationale(result, best, rationale=rationale),
        position_pct=position_pct,
    )
    if rec.status != PENDING:  # pragma: no cover - defensive
        raise RuntimeError(
            f"create_pending_recommendation returned status {rec.status!r}, "
            "expected 'pending'"
        )
    # Evidence goes on the row the human is reviewing; it is NOT a change.
    rec.backtest_json = json.dumps(_evidence(result, top_n))
    session.commit()
    log.info(
        "optimize: proposed param_change %s for %s (%s=%s) pending approval",
        rec.id, result.strategy_name, result.objective, best.objective_value,
    )
    return rec


__all__ = ["KIND", "propose_from_result"]
