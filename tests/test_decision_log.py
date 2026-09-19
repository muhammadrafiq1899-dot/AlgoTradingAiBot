"""Advisory decision log: record -> evaluate -> lessons, plus schema migration.

Everything runs against a seeded SQLite database; no network and no LLM. The
outcomes are asserted against realized (seeded) trades, and the migration test
upgrades an *existing* database in place to prove no data is lost.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from algotrading.ai.decision_log import (
    FLAT,
    LOSS,
    PENDING,
    WIN,
    build_lessons,
    evaluate_due,
    record_applied,
    reflect,
    sync_applied,
)
from algotrading.db import SCHEMA_VERSION, get_session_factory, get_schema_version, init_db
from algotrading.db.models import (
    AIDecisionLog,
    AIRecommendation,
    Candle,
    Strategy,
    Trade,
)

HOUR_MS = 3_600_000


def _candles(symbol: str, closes: list[float], end: datetime) -> list[Candle]:
    """1h candles ending at ``end`` (aligned backwards)."""
    start = end - timedelta(hours=len(closes))
    return [
        Candle(symbol=symbol, interval="1h",
               ts=int((start + timedelta(hours=i)).timestamp() * 1000),
               open=c, high=c, low=c, close=c, volume=1.0)
        for i, c in enumerate(closes)
    ]


def _trade(symbol: str, entry: float, pnl: float, closed: datetime, qty: float = 1.0) -> Trade:
    return Trade(
        symbol=symbol,
        entry_qty=qty,
        entry_avg_price=entry,
        exit_avg_price=entry + (pnl / qty),
        realized_pnl=pnl,
        fees=0.0,
        opened_at=closed - timedelta(hours=2),
        closed_at=closed,
        loss_reasons="[]",
        strategy_id=1,
    )


@pytest.fixture()
def factory(tmp_path):
    db = str(tmp_path / "decision.db")
    init_db(db)
    return get_session_factory(db)


@pytest.fixture()
def session(factory):
    sess = factory()
    sess.add(Strategy(name="ema_crossover", version=1, status="active"))
    sess.commit()
    yield sess
    sess.close()


def _applied(session, *, days_ago: int, status: str = "applied", kind="param_change",
             name="ema_crossover") -> AIRecommendation:
    applied_at = datetime.now(timezone.utc) - timedelta(days=days_ago)
    rec = AIRecommendation(
        kind=kind, strategy_name=name, content_json="{}",
        status=status, rationale="tune", reviewed_at=applied_at,
    )
    session.add(rec)
    session.commit()
    return rec


# --- record -------------------------------------------------------------------

def test_record_applied_creates_one_row_and_is_idempotent(session):
    rec = _applied(session, days_ago=1)
    entry = record_applied(session, rec)
    assert entry is not None
    assert entry.recommendation_id == rec.id
    assert entry.kind == "param_change"
    assert entry.strategy_name == "ema_crossover"
    assert entry.outcome == PENDING
    assert entry.horizon_days == 7

    again = record_applied(session, rec)
    assert again.id == entry.id
    assert session.query(AIDecisionLog).count() == 1


def test_record_applied_ignores_unapplied_recommendations(session):
    rec = _applied(session, days_ago=1, status="pending")
    assert record_applied(session, rec) is None
    assert session.query(AIDecisionLog).count() == 0


def test_sync_applied_backfills_approved_recommendations(session):
    # The apply path lives in another module, so the scheduler observes it here.
    _applied(session, days_ago=30)
    _applied(session, days_ago=20, kind="new_strategy", name="momentum_nudge")
    _applied(session, days_ago=1, status="pending")

    recorded = sync_applied(session)
    assert len(recorded) == 2
    assert sync_applied(session) == []          # idempotent
    assert session.query(AIDecisionLog).count() == 2


# --- evaluate -----------------------------------------------------------------

def _seed_outcome(session, *, days_ago: int, pnl: float, entry: float = 100.0,
                  qty: float = 1.0, benchmark: tuple[float, float] = (100.0, 99.0)):
    rec = _applied(session, days_ago=days_ago)
    entry_row = record_applied(session, rec)
    applied_at = entry_row.applied_at.replace(tzinfo=timezone.utc)
    closed = applied_at + timedelta(days=3)
    session.add(_trade("BTC/USDT", entry, pnl, closed, qty=qty))
    # A benchmark candle at/before the decision and another inside the window,
    # so the move can actually be measured (base=100 -> last=99 => -1%).
    session.add_all(_candles(
        "BTC/USDT", [benchmark[0], benchmark[0], benchmark[1]],
        end=applied_at + timedelta(hours=2),
    ))
    session.commit()
    return entry_row


def test_evaluate_due_marks_a_win(session):
    entry = _seed_outcome(session, days_ago=10, pnl=10.0)  # +10% on a 100 cost

    evaluated = evaluate_due(session, 7, benchmark_symbol="BTC/USDT")

    assert [e.id for e in evaluated] == [entry.id]
    session.refresh(entry)
    assert entry.outcome == WIN
    assert entry.pnl_pct == pytest.approx(10.0)
    assert entry.benchmark_pct == pytest.approx(-1.0)
    assert entry.evaluated_at is not None
    metrics = json.loads(entry.metrics_json)
    assert metrics["n_trades"] == 1
    assert metrics["n_wins"] == 1
    assert metrics["benchmark_symbol"] == "BTC/USDT"


def test_evaluate_due_marks_a_loss(session):
    entry = _seed_outcome(session, days_ago=10, pnl=-5.0)
    evaluate_due(session, 7, benchmark_symbol="BTC/USDT")
    session.refresh(entry)
    assert entry.outcome == LOSS
    assert entry.pnl_pct == pytest.approx(-5.0)


def test_evaluate_due_marks_flat_inside_the_band(session):
    entry = _seed_outcome(session, days_ago=10, pnl=0.01)  # +0.01% < 0.05 band
    evaluate_due(session, 7, benchmark_symbol="BTC/USDT")
    session.refresh(entry)
    assert entry.outcome == FLAT


def test_evaluate_due_with_no_trades_reports_flat_and_says_so(session):
    entry = _seed_outcome(session, days_ago=10, pnl=0.0)
    session.query(Trade).delete()
    session.commit()

    evaluate_due(session, 7, benchmark_symbol="BTC/USDT")

    session.refresh(entry)
    assert entry.outcome == FLAT
    assert entry.pnl_pct == pytest.approx(0.0)
    assert json.loads(entry.metrics_json)["n_trades"] == 0


def test_evaluate_due_skips_a_closed_horizon_not_elapsed(session):
    entry = _seed_outcome(session, days_ago=2, pnl=50.0)

    assert evaluate_due(session, 7, benchmark_symbol="BTC/USDT") == []

    session.refresh(entry)
    assert entry.outcome == PENDING
    assert entry.evaluated_at is None
    assert entry.pnl_pct is None


def test_evaluate_due_without_candles_leaves_the_benchmark_empty(session):
    entry = _seed_outcome(session, days_ago=10, pnl=10.0)
    session.query(Candle).delete()
    session.commit()

    evaluate_due(session, 7, benchmark_symbol="BTC/USDT")

    session.refresh(entry)
    assert entry.outcome == WIN
    assert entry.benchmark_pct is None


def test_evaluate_due_is_idempotent(session):
    _seed_outcome(session, days_ago=10, pnl=10.0)
    assert len(evaluate_due(session, 7, benchmark_symbol="BTC/USDT")) == 1
    # Already filled in: a second run has nothing due.
    assert evaluate_due(session, 7, benchmark_symbol="BTC/USDT") == []


# --- lessons + reflection -----------------------------------------------------

def test_build_lessons_renders_the_expected_line(session):
    entry = _seed_outcome(session, days_ago=10, pnl=-2.1, entry=100.0)
    evaluate_due(session, 7, benchmark_symbol="BTC/USDT")
    session.refresh(entry)

    stamp = entry.applied_at.strftime("%Y-%m-%d")
    lessons = build_lessons(session, limit=5)

    assert lessons == [
        f"{stamp} param_change on ema_crossover: 7d -2.1% vs BTC/USDT -1.0% (loss)"
    ]


def test_build_lessons_is_deterministic_and_capped(session):
    for days in (40, 30, 20):
        _seed_outcome(session, days_ago=days, pnl=5.0)
    evaluate_due(session, 7, benchmark_symbol="BTC/USDT")

    first = build_lessons(session, limit=2)
    second = build_lessons(session, limit=2)
    assert first == second
    assert len(first) == 2
    # Newest decision first.
    newest = max(e.applied_at for e in session.query(AIDecisionLog).all())
    assert first[0].startswith(newest.strftime("%Y-%m-%d"))


def test_build_lessons_empty_without_evaluated_rows(session):
    _seed_outcome(session, days_ago=1, pnl=5.0)   # still inside its horizon
    assert build_lessons(session) == []


def test_reflect_stores_a_paragraph_and_is_stable(session):
    entry = _seed_outcome(session, days_ago=10, pnl=10.0)
    evaluate_due(session, 7, benchmark_symbol="BTC/USDT")

    assert entry.reflection, "evaluate_due should leave a reflection"
    first_text = entry.reflection
    assert "one sample" in first_text
    assert "worked" in first_text

    # Nothing changed, so nothing to rewrite.
    assert reflect(session) == []
    session.refresh(entry)
    assert entry.reflection == first_text

    # Wiping it and re-running reproduces the same deterministic paragraph.
    entry.reflection = ""
    session.commit()
    assert [r.id for r in reflect(session)] == [entry.id]
    assert entry.reflection == first_text


def test_reflection_mentions_the_benchmark_comparison(session):
    entry = _seed_outcome(session, days_ago=10, pnl=10.0)  # +10% vs -1%
    evaluate_due(session, 7, benchmark_symbol="BTC/USDT")
    session.refresh(entry)
    assert "outperformed the benchmark" in entry.reflection


# --- schema migration ---------------------------------------------------------

def test_init_db_adds_the_table_to_an_existing_database(tmp_path):
    db = str(tmp_path / "old.db")
    init_db(db)
    # An existing v2 database with real data in it...
    sess = get_session_factory(db)()
    sess.add(Strategy(name="ema_crossover", version=1, status="active"))
    sess.commit()
    sess.close()

    # ...that predates this feature: drop the new table and rewind the version.
    raw = get_session_factory(db)()
    raw.execute(text("DROP TABLE ai_decision_log"))
    raw.execute(text("UPDATE meta SET value = '2' WHERE key = 'schema_version'"))
    raw.commit()
    raw.close()
    assert get_schema_version(db) == 2

    init_db(db)  # must re-create the table without wiping anything

    assert get_schema_version(db) == SCHEMA_VERSION
    sess = get_session_factory(db)()
    try:
        assert sess.execute(text("SELECT COUNT(*) FROM ai_decision_log")).scalar() == 0
        assert sess.query(Strategy).filter_by(name="ema_crossover").count() == 1
    finally:
        sess.close()


def test_migration_adds_missing_columns_without_losing_rows(tmp_path):
    db = str(tmp_path / "partial.db")
    init_db(db)
    raw = get_session_factory(db)()
    raw.execute(text("DROP TABLE ai_decision_log"))
    # A half-shaped table, as if an earlier build created it.
    raw.execute(text(
        "CREATE TABLE ai_decision_log (id INTEGER PRIMARY KEY, strategy_name VARCHAR(64))"
    ))
    raw.execute(text("INSERT INTO ai_decision_log (id, strategy_name) VALUES (1, 'kept')"))
    raw.execute(text("UPDATE meta SET value = '2' WHERE key = 'schema_version'"))
    raw.commit()
    raw.close()

    init_db(db)

    sess = get_session_factory(db)()
    try:
        cols = {row[1] for row in sess.execute(text("PRAGMA table_info(ai_decision_log)"))}
        assert {"outcome", "pnl_pct", "benchmark_pct", "metrics_json", "reflection",
                "horizon_days", "applied_at", "evaluated_at"} <= cols
        assert sess.execute(text("SELECT strategy_name FROM ai_decision_log")).scalar() == "kept"
    finally:
        sess.close()


def test_init_db_keeps_working_on_a_fresh_database(tmp_path):
    db = str(tmp_path / "fresh.db")
    init_db(db)
    init_db(db)  # idempotent
    assert get_schema_version(db) == SCHEMA_VERSION
    sess = get_session_factory(db)()
    try:
        assert sess.query(AIDecisionLog).count() == 0
    finally:
        sess.close()
