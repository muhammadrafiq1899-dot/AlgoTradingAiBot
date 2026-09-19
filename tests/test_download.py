"""P2-9: history download (resume, row cap, '1M' safety) + on-disk candle files.

Everything here runs offline against a fake venue: the point of the module is
that a phone can pull real depth *resumably* and then replay it from a file
with no network and no database, so the tests must not need either.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from algotrading.backtest.runner import run_backtest
from algotrading.db import get_session, init_db
from algotrading.market.base import Candle, MarketDataProvider
from algotrading.market.candles import INTERVAL_MS, CandleStore
from algotrading.market.download import (
    CANDLE_HEADER,
    CandleFileError,
    download_history,
    export_candles,
    load_candles_file,
)

FIXED_NOW = 1_700_000_000_000  # 2023-11-14T22:13:20Z — deterministic "now"
HOUR = INTERVAL_MS["1h"]


@pytest.fixture()
def store(tmp_path):
    db = str(tmp_path / "download.db")
    init_db(db)
    with get_session(db) as session:
        yield CandleStore(session)


class FakeProvider(MarketDataProvider):
    """Deterministic venue: `count` candles ending at `end_ms`, served per page.

    Records every request so the tests can assert *what* was asked for (a
    resume must ask for the newest stored candle, not the whole window), and
    fails the test instead of hanging if a paging loop never terminates.
    """

    MAX_CALLS = 200

    def __init__(self, count=48, interval_ms=HOUR, end_ms=FIXED_NOW, page=1000):
        self.count = count
        self.interval_ms = interval_ms
        self.page = page
        self.calls: list[tuple[str, str, int | None]] = []
        self.fail_symbols: set[str] = set()
        self.end_ms = end_ms - (end_ms % interval_ms)

    def history(self, symbol: str, interval: str) -> list[Candle]:
        start = self.end_ms - (self.count - 1) * self.interval_ms
        out = []
        for i in range(self.count):
            ts = start + i * self.interval_ms
            o = 100.0 + i
            out.append(Candle(symbol, interval, ts, o, o + 2, o - 2, o + 1, 1.5))
        return out

    def fetch_klines(self, symbol, interval, since_ms=None):
        if len(self.calls) > self.MAX_CALLS:
            raise AssertionError("paging loop did not terminate")
        self.calls.append((symbol, interval, since_ms))
        if symbol in self.fail_symbols:
            raise RuntimeError("venue exploded")
        candles = self.history(symbol, interval)
        if since_ms is not None:
            candles = [c for c in candles if c.ts >= since_ms]
        return candles[: self.page]

    def fetch_ticker_price(self, symbol: str) -> float:
        return 100.0


class MonthlyProvider(FakeProvider):
    """A '1M' venue: candles open on real calendar months (28-31 days long).

    This is where the approximate-interval stepping rule earns its keep: the
    fake, like Binance, only returns candles whose open time is >= the requested
    cursor, so a step longer than the shortest month silently drops one.
    """

    def __init__(self, months: list[int], end_ms: int = FIXED_NOW, page: int = 1000):
        super().__init__(
            count=len(months), interval_ms=INTERVAL_MS["1M"], end_ms=end_ms, page=page
        )
        self._months = sorted(months)

    def history(self, symbol: str, interval: str) -> list[Candle]:
        out = []
        for i, ts in enumerate(self._months):
            o = 100.0 + i
            out.append(Candle(symbol, interval, ts, o, o + 2, o - 2, o + 1, 1.5))
        return out


def _month_starts(end_ms: int, n: int) -> list[int]:
    """`n` month-open timestamps ending at the month containing `end_ms`."""
    end = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)
    year, month = end.year, end.month
    out = []
    for _ in range(n):
        out.append(int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000))
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return sorted(out)


def _stored_ts(store: CandleStore, symbol: str, interval: str) -> list[int]:
    return [row.ts for row in store.get(symbol, interval, limit=100_000)]


# ---------------------------------------------------------------------------
# download_history
# ---------------------------------------------------------------------------

def test_full_window_downloads_every_candle(store):
    provider = FakeProvider(count=48)
    report = download_history(
        provider, store, ["BTC/USDT"], ["1h"], 2, now_ms=FIXED_NOW
    )

    assert report["total_rows"] == 48
    assert report["truncated"] is False
    assert report["errors"] == 0
    (entry,) = report["results"]
    assert entry["symbol"] == "BTC/USDT"
    assert entry["interval"] == "1h"
    assert entry["rows"] == 48
    assert entry["latest_before"] is None
    assert entry["resumed"] is False
    assert entry["ok"] is True
    assert len(_stored_ts(store, "BTC/USDT", "1h")) == 48


def test_rerun_resumes_from_the_newest_stored_candle(store):
    provider = FakeProvider(count=48)
    download_history(provider, store, ["BTC/USDT"], ["1h"], 2, now_ms=FIXED_NOW)
    latest = max(_stored_ts(store, "BTC/USDT", "1h"))
    calls_before = len(provider.calls)

    report = download_history(
        provider, store, ["BTC/USDT"], ["1h"], 2, now_ms=FIXED_NOW
    )

    (entry,) = report["results"]
    assert entry["resumed"] is True
    assert entry["latest_before"] == latest
    # Only the still-forming newest candle is fetched again (its values can
    # change until the bar closes) — not the whole two-day window.
    assert report["total_rows"] == 1
    assert _stored_ts(store, "BTC/USDT", "1h") == sorted(
        _stored_ts(store, "BTC/USDT", "1h")
    )
    assert len(_stored_ts(store, "BTC/USDT", "1h")) == 48
    # The venue was asked from the newest stored candle, not from the window.
    assert provider.calls[calls_before][2] == latest


def test_resume_after_gap_older_than_the_window_refetches(store):
    """Stored data that is older than the requested window is not a resume."""
    provider = FakeProvider(count=48)
    download_history(provider, store, ["BTC/USDT"], ["1h"], 2, now_ms=FIXED_NOW)

    # A fresh window ending "later" than the stored data: nothing of the new
    # window is on disk, so the whole window has to be fetched.
    later = FIXED_NOW + 10 * 24 * HOUR
    report = download_history(
        provider, store, ["BTC/USDT"], ["1h"], 2, now_ms=later
    )
    (entry,) = report["results"]
    assert entry["resumed"] is False
    assert entry["latest_before"] < entry["start_ts"]


def test_row_cap_is_a_hard_ceiling_and_marks_capped_pairs(store):
    provider = FakeProvider(count=48)
    report = download_history(
        provider, store, ["BTC/USDT", "ETH/USDT"], ["1h"], 2,
        max_rows=5, now_ms=FIXED_NOW,
    )

    assert report["total_rows"] == 5
    assert report["truncated"] is True
    first, second = report["results"]
    assert first["rows"] == 5 and first["capped"] is True
    # The second pair got nothing because the budget was spent, and it says so
    # instead of silently pretending the download succeeded.
    assert second["rows"] == 0 and second["capped"] is True
    assert len(_stored_ts(store, "ETH/USDT", "1h")) == 0


def test_capped_run_resumes_and_completes_the_window(store):
    provider = FakeProvider(count=48)
    download_history(
        provider, store, ["BTC/USDT"], ["1h"], 2, max_rows=5, now_ms=FIXED_NOW
    )
    assert len(_stored_ts(store, "BTC/USDT", "1h")) == 5

    download_history(
        provider, store, ["BTC/USDT"], ["1h"], 2, max_rows=1000, now_ms=FIXED_NOW
    )
    assert len(_stored_ts(store, "BTC/USDT", "1h")) == 48


def test_progress_callback_reports_each_pair(store):
    provider = FakeProvider(count=48)
    seen: list[tuple[str, str, int]] = []
    download_history(
        provider, store, ["BTC/USDT"], ["1h", "4h"], 2,
        now_ms=FIXED_NOW, progress=lambda s, i, n: seen.append((s, i, n)),
    )
    assert [call[:2] for call in seen] == [("BTC/USDT", "1h"), ("BTC/USDT", "4h")]
    assert seen[0][2] == 48


def test_one_failing_pair_does_not_block_the_others(store):
    provider = FakeProvider(count=48)
    provider.fail_symbols.add("ETH/USDT")
    report = download_history(
        provider, store, ["BTC/USDT", "ETH/USDT"], ["1h"], 2, now_ms=FIXED_NOW
    )

    assert report["errors"] == 1
    ok, failed = report["results"]
    assert ok["ok"] is True and ok["rows"] == 48
    assert failed["ok"] is False
    assert "venue exploded" in failed["error"]
    # The good pair is still fully stored, so a re-run only retries the bad one.
    assert len(_stored_ts(store, "BTC/USDT", "1h")) == 48


def test_bad_arguments_are_rejected_clearly(store):
    provider = FakeProvider()
    with pytest.raises(ValueError, match="at least one symbol"):
        download_history(provider, store, [], ["1h"], 2)
    with pytest.raises(ValueError, match="at least one interval"):
        download_history(provider, store, ["BTC/USDT"], [], 2)
    with pytest.raises(ValueError, match="unknown interval"):
        download_history(provider, store, ["BTC/USDT"], ["3h"], 2)
    with pytest.raises(ValueError, match="days must be"):
        download_history(provider, store, ["BTC/USDT"], ["1h"], 0)
    with pytest.raises(ValueError, match="max_rows"):
        download_history(provider, store, ["BTC/USDT"], ["1h"], 2, max_rows=0)


def test_approximate_month_interval_never_skips_a_candle(store):
    """'1M' is a 30-day approximation; the paging step must not skip a month.

    `page=1` makes the venue hand back one candle per request, so what is
    actually under test is the *cursor step* (not how much the venue volunteers
    at once): the fake — like the real API — only returns candles at/after the
    cursor. Stepping a flat 30 days from 2023-02-01 lands on 2023-03-03 and
    drops the March candle; stepping the shortest month (28 days) lands on
    2023-03-01 and keeps it.
    """
    months = _month_starts(FIXED_NOW, 14)  # 2022-10 .. 2023-11
    provider = MonthlyProvider(months, page=1)

    report = download_history(
        provider, store, ["BTC/USDT"], ["1M"], 365, now_ms=FIXED_NOW
    )

    window_start = FIXED_NOW - 365 * 86_400_000
    aligned = window_start - (window_start % INTERVAL_MS["1M"])
    expected = [ts for ts in months if aligned <= ts <= FIXED_NOW]
    assert len(expected) >= 12
    assert _stored_ts(store, "BTC/USDT", "1M") == expected
    assert report["total_rows"] == len(expected)
    # And it terminates: one request per candle plus the empty catch-up call.
    assert len(provider.calls) <= len(expected) + 2


def test_month_interval_row_cap_still_terminates(store):
    provider = MonthlyProvider(_month_starts(FIXED_NOW, 14))
    report = download_history(
        provider, store, ["BTC/USDT"], ["1M"], 365, max_rows=3, now_ms=FIXED_NOW
    )
    assert report["total_rows"] == 3
    assert report["truncated"] is True


# ---------------------------------------------------------------------------
# candle files
# ---------------------------------------------------------------------------

def _sample_candles(n: int = 5) -> list[Candle]:
    out = []
    for i in range(n):
        ts = FIXED_NOW + i * HOUR
        out.append(Candle("BTC/USDT", "1h", ts, 100.0 + i, 102.5 + i, 99.0 + i,
                          101.25 + i, 3.5 + i))
    return out


def test_csv_round_trip_is_exact(tmp_path):
    path = tmp_path / "candles.csv"
    candles = _sample_candles()
    assert export_candles(candles, path) == 5

    assert path.read_text().splitlines()[0] == CANDLE_HEADER
    loaded = load_candles_file(path)
    assert loaded == candles  # exact: floats use shortest round-trip repr


def test_jsonl_round_trip_is_exact(tmp_path):
    path = tmp_path / "candles.jsonl"
    candles = _sample_candles()
    assert export_candles(candles, path) == 5
    loaded = load_candles_file(path)
    assert loaded == candles


def test_load_sorts_oldest_first(tmp_path):
    path = tmp_path / "candles.jsonl"
    export_candles(list(reversed(_sample_candles())), path)
    loaded = load_candles_file(path)
    assert [c.ts for c in loaded] == sorted(c.ts for c in loaded)


def test_export_accepts_orm_rows_and_dicts(tmp_path):
    candles = _sample_candles()
    as_dicts = [c.as_dict() for c in candles]
    path = tmp_path / "dicts.csv"
    export_candles(as_dicts, path)
    assert load_candles_file(path) == candles


def test_export_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "candles.csv"
    export_candles(_sample_candles(3), path)
    export_candles(_sample_candles(5), path)  # rewrite
    assert len(load_candles_file(path)) == 5
    assert [p.name for p in tmp_path.iterdir()] == ["candles.csv"]


def test_export_creates_missing_parent_directory(tmp_path):
    path = tmp_path / "nested" / "candles.jsonl"
    export_candles(_sample_candles(2), path)
    assert path.exists()


def test_unsupported_suffix_is_refused(tmp_path):
    with pytest.raises(CandleFileError, match="unsupported candle file"):
        export_candles(_sample_candles(1), tmp_path / "candles.tsv")
    with pytest.raises(CandleFileError, match="unsupported candle file"):
        load_candles_file(tmp_path / "candles.tsv")


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_candles_file(tmp_path / "nope.csv")


def test_malformed_csv_rows_report_the_line(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        CANDLE_HEADER + "\n"
        + "BTC/USDT,1h,1700000000000,1,2,0.5,1.5,10\n"
        + "BTC/USDT,1h,1700000000000,1,2,0.5,not-a-number,10\n"
    )
    with pytest.raises(CandleFileError) as exc:
        load_candles_file(path)
    assert "line 3" in str(exc.value)
    assert "close" in str(exc.value)


def test_csv_bad_header_and_column_count_are_refused(tmp_path):
    wrong_header = tmp_path / "wrong.csv"
    wrong_header.write_text("ts,open,high,low,close,volume,symbol,interval\n")
    with pytest.raises(CandleFileError, match="bad header"):
        load_candles_file(wrong_header)

    short = tmp_path / "short.csv"
    short.write_text(CANDLE_HEADER + "\nBTC/USDT,1h,1700000000000,1,2\n")
    with pytest.raises(CandleFileError, match="expected 8 columns, got 5"):
        load_candles_file(short)


def test_empty_csv_and_empty_symbol_are_refused(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(CandleFileError, match="empty candle file"):
        load_candles_file(empty)

    blank_symbol = tmp_path / "blank.csv"
    blank_symbol.write_text(
        CANDLE_HEADER + "\n,1h,1700000000000,1,2,0.5,1.5,10\n"
    )
    with pytest.raises(CandleFileError, match="symbol"):
        load_candles_file(blank_symbol)


def test_malformed_jsonl_rows_report_the_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    good = {"symbol": "BTC/USDT", "interval": "1h", "ts": FIXED_NOW,
            "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}
    path.write_text(
        json.dumps(good) + "\n"
        + "{not json}\n"
    )
    with pytest.raises(CandleFileError) as exc:
        load_candles_file(path)
    assert "line 2" in str(exc.value)
    assert "invalid JSON" in str(exc.value)

    missing = tmp_path / "missing.jsonl"
    incomplete = {k: v for k, v in good.items() if k != "volume"}
    missing.write_text(json.dumps(incomplete) + "\n")
    with pytest.raises(CandleFileError, match="volume"):
        load_candles_file(missing)

    not_object = tmp_path / "notobj.jsonl"
    not_object.write_text("[1, 2, 3]\n")
    with pytest.raises(CandleFileError, match="expected a JSON object"):
        load_candles_file(not_object)


def test_exported_candles_feed_run_backtest_unchanged(tmp_path):
    """The whole point of the file format: replay with no DB and no network."""
    candles = [
        Candle("BTC/USDT", "1h", FIXED_NOW + i * HOUR, 100.0 + i, 101.0 + i,
               99.0 + i, 100.5 + i, 1.0 + i * 0.01)
        for i in range(220)
    ]
    path = tmp_path / "history.csv"
    export_candles(candles, path)
    loaded = load_candles_file(path)

    result = run_backtest(
        loaded, "ema_crossover", {"fast_period": 5, "slow_period": 20, "position_pct": 0.2}
    )
    assert result.symbol == "BTC/USDT"
    assert result.interval == "1h"
    assert result.n_trades >= 0
