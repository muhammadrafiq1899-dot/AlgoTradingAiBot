"""Analytics service: read the trade ledger, compute metrics, persist summaries.

Periodic job (every 30m + daily, per ScheduleConfig) snapshots closed trades into
analytics_summaries rows so the AI prompt builder and /summary have cheap,
deterministic inputs — no live ledger scan needed at query time.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from algotrading.db.models import AnalyticsSummary, Trade

log = logging.getLogger(__name__)


class AnalyticsService:
    def __init__(self, session, period: str = "30m", lookback_minutes: int = 30):
        self._session = session
        self._period = period
        self._lookback = timedelta(minutes=lookback_minutes)

    def run(self, strategy_id: int | None = None) -> AnalyticsSummary:
        """Compute metrics for trades since `lookback` and persist a summary."""
        from algotrading.analytics.metrics import compute_metrics

        since = datetime.now(timezone.utc) - self._lookback
        stmt = select(Trade).where(Trade.closed_at >= since)
        if strategy_id is not None:
            stmt = stmt.where(Trade.strategy_id == strategy_id)
        trades = self._session.execute(stmt).scalars().all()

        m = compute_metrics(trades, period=self._period, strategy_id=strategy_id)
        summary = AnalyticsSummary(
            period=self._period,
            strategy_id=strategy_id,
            symbol="ALL",
            metrics_json=json.dumps(m.to_dict()),
        )
        self._session.add(summary)
        self._session.commit()
        log.info("analytics %s: %d trades, win_rate=%s",
                 self._period, m.n_trades, m.win_rate)
        return summary

    def run_daily(self) -> AnalyticsSummary:
        """Full-day summary (since midnight UTC)."""
        from algotrading.analytics.metrics import compute_metrics

        now = datetime.now(timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        trades = self._session.execute(
            select(Trade).where(Trade.closed_at >= midnight)
        ).scalars().all()

        m = compute_metrics(trades, period="daily")
        summary = AnalyticsSummary(
            period="daily", strategy_id=None, symbol="ALL",
            metrics_json=json.dumps(m.to_dict()),
        )
        self._session.add(summary)
        self._session.commit()
        return summary
