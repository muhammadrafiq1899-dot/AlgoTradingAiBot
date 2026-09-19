"""Data depth + portability: bulk history download and on-disk candle files.

Two related jobs that both exist because of where this bot runs (a phone):

* **Depth.** :func:`download_history` pages history into the candle table for a
  set of symbols/intervals, resumably and under a row cap. It is the same code
  path the live tick uses for its incremental fetch (``backfill`` -> batched
  ``upsert``), so a manual download and the running bot cannot disagree about
  what a stored candle looks like.
* **Portability.** :func:`export_candles` / :func:`load_candles_file` move
  candles to and from a plain file (CSV or JSONL) so a backtest can run with no
  network and no database — e.g. on a laptop, or on this phone while the bot is
  stopped. ``load_candles_file`` returns ``algotrading.market.base.Candle``
  objects, which is exactly what ``algotrading.backtest.runner.run_backtest``
  consumes, so no adapter layer is needed anywhere.

Nothing here writes outside the path it is given, and every write is atomic
(temp file + rename): a download killed by Android mid-write must not leave a
half-file that later loads as garbage.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from algotrading.market.base import Candle, MarketDataProvider
from algotrading.market.candles import (
    INTERVAL_MS,
    CandleStore,
    backfill,
)

log = logging.getLogger(__name__)

# The on-disk header, in this order. The CSV writer emits exactly this line and
# the CSV reader refuses anything else, so a file is never half-understood
# (a silently mismatched column order would corrupt every backtest built on it).
CANDLE_COLUMNS: tuple[str, ...] = (
    "symbol", "interval", "ts", "open", "high", "low", "close", "volume",
)
CANDLE_HEADER = ",".join(CANDLE_COLUMNS)

# Suffixes we can read/write. Chosen by suffix, never guessed.
CSV_SUFFIXES = (".csv",)
JSONL_SUFFIXES = (".jsonl", ".ndjson")

# Row cap for one download run. 200k rows is roughly 6 years of hourly candles
# for one pair: enough for real research, small enough that a phone's SQLite
# file stays in the low tens of MB instead of filling the device.
DEFAULT_MAX_ROWS = 200_000

MS_PER_DAY = 86_400_000


class CandleFileError(ValueError):
    """A candle file is malformed, unreadable, or has an unsupported suffix.

    Subclasses ValueError so callers can catch the generic type; the message
    always names the file and (for row problems) the line number, because a
    malformed row is only actionable if you know where it is.
    """


# ---------------------------------------------------------------------------
# history download
# ---------------------------------------------------------------------------

def download_history(
    provider: MarketDataProvider,
    store: CandleStore,
    symbols: Iterable[str],
    intervals: Iterable[str],
    days: int,
    *,
    max_rows: int | None = DEFAULT_MAX_ROWS,
    progress: Callable[[str, str, int], None] | None = None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Download `days` of history for every symbol x interval into `store`.

    Resumable by construction: for each pair the newest stored candle
    (``CandleStore.latest_ts``) becomes the fetch start, so a re-run only asks
    the venue for what is missing (plus the still-forming newest candle, which
    is re-fetched on purpose because it changes until the bar closes). A run
    against an empty table is therefore a full backfill; a second run is a
    nearly-free top-up.

    The row cap (`max_rows`) is a disk guard, not a fetch limit: it is spent
    *per stored row* across the whole run, and once it runs out the remaining
    pairs are reported as `capped` with zero rows instead of being silently
    dropped. Re-run later — resume picks up exactly where it stopped.

    Args:
        provider: market data source.
        store: candle store to write into.
        symbols: e.g. ``["BTC/USDT", "ETH/USDT"]``.
        intervals: subset of ``algotrading.market.candles.INTERVAL_MS``.
        days: history window, counted back from now.
        max_rows: total rows this run may store (None = unlimited).
        progress: called as ``progress(symbol, interval, rows_stored)`` once
            per pair, so a CLI can show the pair that just finished.
        now_ms: epoch-ms "now" override (tests; default: the wall clock).

    Returns:
        A JSON-serializable report::

            {
              "days": 365, "max_rows": 200000, "total_rows": 1234,
              "truncated": false, "errors": 0,
              "results": [
                {"symbol": "BTC/USDT", "interval": "1h", "rows": 1234,
                 "start_ts": 1700000000000, "latest_before": 1699000000000,
                 "resumed": true, "capped": false, "ok": true,
                 "error": null},
                ...
              ]
            }

    Raises:
        ValueError: no symbols/intervals, an unknown interval, or days < 1.
    """
    symbol_list = [s for s in (str(s).strip() for s in symbols) if s]
    interval_list = [i for i in (str(i).strip() for i in intervals) if i]
    if not symbol_list:
        raise ValueError("download_history needs at least one symbol")
    if not interval_list:
        raise ValueError("download_history needs at least one interval")
    unknown = [i for i in interval_list if i not in INTERVAL_MS]
    if unknown:
        raise ValueError(
            f"unknown interval(s) {unknown}; known: {sorted(INTERVAL_MS)}"
        )
    if days < 1:
        raise ValueError("days must be >= 1")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be >= 1 or None")

    now = int(now_ms if now_ms is not None else time.time() * 1000)
    window_start = now - days * MS_PER_DAY

    results: list[dict[str, Any]] = []
    total_rows = 0
    errors = 0

    for symbol in symbol_list:
        for interval in interval_list:
            latest_before = store.latest_ts(symbol, interval)
            # Resume when we already hold data inside the requested window;
            # otherwise (empty table, or data older than the window) the whole
            # window has to be fetched.
            resumed = latest_before is not None and latest_before >= window_start
            start_ms = max(latest_before, window_start) if resumed else window_start

            remaining = None if max_rows is None else max_rows - total_rows
            rows = 0
            capped = False
            error: str | None = None
            if remaining is not None and remaining <= 0:
                capped = True  # budget spent by earlier pairs; resumable later
            else:
                try:
                    rows = backfill(
                        provider,
                        store,
                        symbol,
                        interval,
                        days,
                        since_ms=start_ms,
                        max_rows=remaining,
                    )
                    capped = remaining is not None and rows >= remaining
                except Exception as exc:  # noqa: BLE001 - network/venue boundary
                    # One flaky pair must not cost the user the other pairs.
                    # Whatever pages already committed stay in the table, and
                    # the next run resumes from there.
                    log.error(
                        "download failed for %s %s: %s", symbol, interval, exc
                    )
                    error = str(exc)
                    errors += 1
                else:
                    log.info(
                        "download %s %s: %s rows (%s)",
                        symbol, interval, rows,
                        "resumed" if resumed else "full window",
                    )

            total_rows += rows
            results.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "rows": rows,
                    "start_ts": start_ms,
                    "latest_before": latest_before,
                    "resumed": resumed,
                    "capped": capped,
                    "ok": error is None,
                    "error": error,
                }
            )
            if progress is not None:
                progress(symbol, interval, rows)

    return {
        "days": days,
        "max_rows": max_rows,
        "total_rows": total_rows,
        "truncated": max_rows is not None and total_rows >= max_rows,
        "errors": errors,
        "results": results,
    }


# ---------------------------------------------------------------------------
# candle files (CSV / JSONL)
# ---------------------------------------------------------------------------

def export_candles(rows: Sequence[Any], path: str | Path) -> int:
    """Write candles to `path`, format chosen by suffix (.csv / .jsonl).

    `rows` may be `Candle` dataclasses or the ORM `Candle` rows from
    ``CandleStore.get`` — both expose the same eight fields, which is the whole
    point of the flat on-disk shape.

    The write is atomic (temp file in the same directory, then ``os.replace``):
    a process killed mid-write leaves the old file untouched rather than a
    truncated one, and a reader never observes a partial file. Nothing is
    written outside `path` (the temp file sits next to it and is renamed onto
    it).

    CSV is header-first, columns exactly ``symbol,interval,ts,open,high,low,
    close,volume``. JSONL is one JSON object per line with those same keys.
    Floats are written with Python's shortest round-trip repr, so a
    export -> load round trip compares exactly equal.

    Args:
        rows: candle-like objects (attribute access, or dicts).
        path: destination ending in .csv, .jsonl or .ndjson.

    Returns:
        Number of rows written.

    Raises:
        CandleFileError: unsupported suffix or a row missing a field.
    """
    target = Path(path)
    kind = _kind_for(target)
    # Validate every row *before* opening the file: a bad row must not leave a
    # partial export behind (nor overwrite a good earlier one).
    records = [_record(row, index) for index, row in enumerate(rows)]

    if kind == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(CANDLE_COLUMNS)
        for rec in records:
            writer.writerow([rec[c] for c in CANDLE_COLUMNS])
        text = buf.getvalue()
    else:
        lines = [json.dumps(rec, sort_keys=True) for rec in records]
        text = "".join(line + "\n" for line in lines)

    _write_atomic(target, text)
    log.info("exported %s candles to %s", len(records), target)
    return len(records)


def load_candles_file(path: str | Path) -> list[Candle]:
    """Read a candle file written by :func:`export_candles` (or by hand).

    Returns `Candle` objects sorted oldest -> newest (by ``ts``), which is the
    order ``algotrading.backtest.runner.run_backtest`` requires — so the file
    can be replayed directly with no adapter and no database.

    Args:
        path: a .csv, .jsonl or .ndjson file.

    Returns:
        Candles, oldest first.

    Raises:
        CandleFileError: unsupported suffix, missing/rewritten CSV header,
            malformed row (wrong column count, non-numeric number, empty
            symbol/interval, missing JSON key) — the message names the file and
            line so the row can actually be found.
        FileNotFoundError: the file does not exist.
    """
    target = Path(path)
    kind = _kind_for(target)
    if not target.exists():
        raise FileNotFoundError(f"candle file not found: {target}")

    if kind == "csv":
        candles = _load_csv(target)
    else:
        candles = _load_jsonl(target)
    candles.sort(key=lambda c: c.ts)
    return candles


# --- internals -------------------------------------------------------------

def _kind_for(path: Path) -> str:
    """Map a file suffix to a format, refusing anything else.

    Suffix-based on purpose: the user's `--export data/candles.csv` decides the
    format, and a typo like `.tsv` fails loudly instead of writing CSV into a
    file nobody can parse.
    """
    suffix = path.suffix.lower()
    if suffix in CSV_SUFFIXES:
        return "csv"
    if suffix in JSONL_SUFFIXES:
        return "jsonl"
    raise CandleFileError(
        f"unsupported candle file {str(path)!r}: expected one of "
        f"{list(CSV_SUFFIXES + JSONL_SUFFIXES)}"
    )


def _record(row: Any, index: int) -> dict[str, Any]:
    """Normalize a Candle dataclass / ORM row / dict into a plain record."""
    out: dict[str, Any] = {}
    for col in CANDLE_COLUMNS:
        if isinstance(row, dict):
            if col not in row:
                raise CandleFileError(f"row {index}: missing field {col!r}")
            value = row[col]
        else:
            try:
                value = getattr(row, col)
            except AttributeError as exc:
                raise CandleFileError(
                    f"row {index}: not a candle (no {col!r} field)"
                ) from exc
        if col == "ts":
            out[col] = int(value)
        elif col in ("symbol", "interval"):
            out[col] = str(value)
        else:
            out[col] = float(value)
    return out


def _write_atomic(path: Path, text: str) -> None:
    """Write `text` to `path` via a temp file in the same directory + rename.

    Same-directory matters: ``os.replace`` is only atomic within one filesystem,
    and on Android the app's data dir and the CWD can be different mounts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:  # pragma: no cover - best-effort cleanup
                log.warning("could not remove temp file %s", tmp)


def _parse_record(where: str, raw: dict[str, Any]) -> Candle:
    """Build a Candle from raw fields, raising CandleFileError with context."""
    for col in ("symbol", "interval"):
        value = raw.get(col)
        if value is None or str(value).strip() == "":
            raise CandleFileError(f"{where}: missing or empty {col!r}")
    try:
        ts = int(raw["ts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CandleFileError(f"{where}: invalid ts {raw.get('ts')!r}") from exc

    numbers: dict[str, float] = {}
    for col in ("open", "high", "low", "close", "volume"):
        try:
            numbers[col] = float(raw[col])
        except (KeyError, TypeError, ValueError) as exc:
            raise CandleFileError(f"{where}: invalid {col} {raw.get(col)!r}") from exc

    return Candle(
        symbol=str(raw["symbol"]).strip(),
        interval=str(raw["interval"]).strip(),
        ts=ts,
        open=numbers["open"],
        high=numbers["high"],
        low=numbers["low"],
        close=numbers["close"],
        volume=numbers["volume"],
    )


def _load_csv(path: Path) -> list[Candle]:
    """Read the documented CSV shape. The header is validated, not skipped."""
    candles: list[Candle] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    # Skip empty lines (a trailing newline, or a blank separator line).
    rows = [
        (lineno, line)
        for lineno, line in enumerate(lines, start=1)
        if line.strip()
    ]
    if not rows:
        raise CandleFileError(f"{path}: empty candle file (expected header {CANDLE_HEADER})")

    header_line, header_raw = rows[0]
    header = [c.strip() for c in next(csv.reader([header_raw]))]
    if header != list(CANDLE_COLUMNS):
        raise CandleFileError(
            f"{path}: line {header_line}: bad header {header!r}; "
            f"expected {CANDLE_HEADER}"
        )

    for lineno, line in rows[1:]:
        fields = next(csv.reader([line]))
        if len(fields) != len(CANDLE_COLUMNS):
            raise CandleFileError(
                f"{path}: line {lineno}: expected {len(CANDLE_COLUMNS)} columns, "
                f"got {len(fields)}"
            )
        raw = dict(zip(CANDLE_COLUMNS, fields))
        candles.append(_parse_record(f"{path}: line {lineno}", raw))
    return candles


def _load_jsonl(path: Path) -> list[Candle]:
    """Read one JSON object per line; blank lines are tolerated."""
    candles: list[Candle] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CandleFileError(f"{path}: line {lineno}: invalid JSON ({exc.msg})") from exc
        if not isinstance(raw, dict):
            raise CandleFileError(
                f"{path}: line {lineno}: expected a JSON object, got {type(raw).__name__}"
            )
        candles.append(_parse_record(f"{path}: line {lineno}", raw))
    return candles
