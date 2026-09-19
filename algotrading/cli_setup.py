"""First-run configuration helpers for the AlgoTrading bot.

Shared by the interactive setup wizard (scripts/setup.py) and the `algobot`
command. All logic lives here so there is a single source of truth for how
secrets are read from / written to .env, and for how the bot is started /
stopped / inspected on the terminal.

Commands (all argparse-free, in the plain terminal style the rest of this file
uses):

    algobot                    interactive menu
    algobot start [args]       start the bot in the background
    algobot stop               stop the bot
    algobot status             show current state
    algobot logs [N]           show the last N log lines
    algobot setup              first-time configuration
    algobot download [...]     fetch history into the DB (resumable, capped)
    algobot export [...]       write trades / equity / summary files from the DB

`download` and `export` are thin wrappers over
``algotrading.market.download`` and ``algotrading.store.export`` so the CLI and
the API/scripts cannot drift apart.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
RUN_BOT = PROJECT_ROOT / "scripts" / "run_bot.sh"
PID_PATH = PROJECT_ROOT / "data" / "algobot.pid"
LOG_PATH = PROJECT_ROOT / "logs" / "algotrading.log"
# How long `algobot stop` waits for a clean shutdown before SIGKILLing the group.
STOP_GRACE_SECONDS = 20

# How many stored candles one `algobot download --export` writes per pair.
# Sized so an export of "everything I just downloaded" fits in one pass without
# a phone-sized memory spike.
EXPORT_LIMIT = 100_000

DOWNLOAD_HELP = """\
algobot download — fetch candles into the local database.

Usage:
  algobot download [--symbols A,B] [--intervals 1h,4h] [--days N]
                   [--max-rows N] [--export PATH] [--db PATH]

Options:
  --symbols    comma-separated pairs       (default: market.symbols)
  --intervals  comma-separated intervals   (default: market.intervals)
  --days       history window in days      (default: market.backfill_days)
  --max-rows   total rows this run may store (default: 200000)
  --export     also write the stored candles to a .csv or .jsonl file
  --db         database file to write into (default: the configured one)

Re-running is cheap and safe: only the candles that are missing are fetched,
and an interrupted run resumes where it stopped.
"""

EXPORT_HELP = """\
algobot export — write the bot's trades, equity curve and summary to files.

Usage:
  algobot export [--dir DIR]
                 [--trades-csv PATH] [--trades-json PATH]
                 [--equity-csv PATH] [--summary-json PATH]

Options:
  --dir            write all four files into this directory
  --trades-csv     trades as CSV
  --trades-json    trades as a JSON array
  --equity-csv     equity curve (closed trades, in time order)
  --summary-json   counts, totals, latest analytics, mode/strategy metadata
  --db             database file to read (default: the configured one)

Read-only: exports never change bot state.
"""

# Field order shown to the user during setup.
ENV_FIELDS: list[dict] = [
    {
        "key": "TELEGRAM_BOT_TOKEN",
        "label": "Telegram bot token",
        "help": "From @BotFather in Telegram (looks like 123456:ABC...). "
                "This is how the bot talks to you.",
        "secret": True,
    },
    {
        "key": "TELEGRAM_ALLOWED_USERS",
        "label": "Your Telegram user ID(s)",
        "help": "Your numeric Telegram ID (ask @userinfobot). Comma-separated "
                "for multiple users. Only these IDs can control the bot.",
        "secret": False,
    },
    {
        "key": "AI_API_KEY",
        "label": "AI assistant API key",
        "help": "Optional. Enables the daily AI strategy suggestions. "
                "Leave blank to run without the AI assistant.",
        "secret": True,
        "optional": True,
    },
    {
        "key": "AI_BASE_URL",
        "label": "AI base URL",
        "help": "OpenAI-compatible endpoint. Default works for OpenAI; change "
                "it for other providers.",
        "secret": False,
        "optional": True,
        "default": "https://api.openai.com/v1",
    },
    {
        "key": "AI_MODEL",
        "label": "AI model",
        "help": "Model name to use for suggestions.",
        "secret": False,
        "optional": True,
        "default": "gpt-4o-mini",
    },
]


# ---------------------------------------------------------------------------
# .env read / write
# ---------------------------------------------------------------------------

def read_env() -> dict[str, str]:
    """Parse the .env file into {KEY: value}. Missing file -> {}."""
    if not ENV_PATH.exists():
        return {}
    values: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip()
    return values


def write_env(values: dict[str, str]) -> None:
    """Write a .env file preserving the original comments/order where possible,
    updating any keys present in `values`. Keys not present keep old values."""
    existing = read_env()
    merged = {**existing, **values}

    # Preserve header + any non-secret comments from the template.
    header = []
    if ENV_PATH.exists():
        src = ENV_PATH.read_text(encoding="utf-8")
        header = [l for l in src.splitlines()
                  if l.strip().startswith("#") or (l.strip() and "=" not in l)]

    lines = list(header)
    if lines and not lines[-1].startswith("#"):
        lines.append("")
    for key in sorted(merged):
        lines.append(f"{key}={merged[key]}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# interactive prompts
# ---------------------------------------------------------------------------

def _mask(value: str) -> str:
    """Shorten a secret for display, e.g. 1234...wxyz."""
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "..." + value[-4:]


def _input(prompt: str, secret: bool = False) -> str:
    """Read a line from stdin. Always plain input() so long-press paste works.

    getpass() is NOT used for secrets: on Termux it opens /dev/tty directly and
    reads there, bypassing stdin — pasted tokens (inserted into stdin) would
    never arrive. Echo of the value is unavoidable in the Termux terminal.
    """
    return input(prompt)


def prompt_for_fields(existing: dict[str, str]) -> dict[str, str]:
    """Ask the user for each field, showing current value and letting them
    keep it by pressing Enter. Returns a dict of changed values."""
    print()
    print("AlgoTrading first-time setup")
    print("============================")
    print("Press Enter to keep the current value shown in brackets. "
          "Type 'skip' to leave a value blank.")
    print()

    updates: dict[str, str] = {}

    for field in ENV_FIELDS:
        key = field["key"]
        current = existing.get(key, "")
        shown = _mask(current) if (field.get("secret") and current) else current
        if not current and field.get("default"):
            shown = field["default"]

        print(f"-- {field['label']} --")
        print(f"   {field['help']}")
        prompt = f"   [{shown}]> "
        while True:
            raw = _input(prompt, secret=bool(field.get("secret"))).strip()
            if raw == "":
                # keep current (or default if none set)
                new = current or field.get("default", "")
                break
            if raw.lower() == "skip":
                new = ""
                break
            # non-secret typed value overrides default too
            new = raw
            break

        updates[key] = new

    return updates


def validate_telegram(token: str) -> bool:
    """Call Telegram's getMe to confirm the token is valid. Returns True on
    success, False on any failure (network or bad token)."""
    if not token:
        return False
    import urllib.request
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            import json
            data = json.loads(resp.read().decode())
            return bool(data.get("ok"))
    except Exception:
        return False


def apply_updates(updates: dict[str, str]) -> None:
    """Persist the setup fields into .env, then init the DB."""
    write_env(updates)
    print("\nSaved .env")
    # Keep AI disabled until a key is actually present.
    # Initialize the database so first run has tables ready.
    _run([str(PYTHON), str(PROJECT_ROOT / "scripts" / "init_db.py")],
         "initializing database")


# ---------------------------------------------------------------------------
# bot lifecycle (used by algobot command)
# ---------------------------------------------------------------------------

def _run(cmd: list[str], what: str) -> int:
    print(f"==> {what} ...")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"!! {what} failed (exit {proc.returncode})")
    return proc.returncode


def get_pid() -> int | None:
    try:
        return int(PID_PATH.read_text().strip())
    except Exception:
        return None


def is_running() -> bool:
    pid = get_pid()
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # Check if process is a zombie (state 'Z' in /proc/<pid>/stat)
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            stat = f.read()
        # stat format: pid (comm) state ...
        # state is the 3rd field
        state = stat.split()[2]
        if state == "Z":
            return False
    except Exception:
        pass
    return True


def start(args: list[str]) -> None:
    if is_running():
        print(f"Bot is already running (pid {get_pid()}).")
        return
    if not PYTHON.exists():
        print("No virtualenv found. Run:  bash scripts/setup_termux.sh")
        return

    # Launch the supervisor loop detached from this terminal.
    log = open(LOG_PATH, "a", encoding="utf-8") if LOG_PATH.parent.exists() else subprocess.DEVNULL
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_PATH, "a", encoding="utf-8")
    devnull = open(os.devnull, "w")
    cmd = ["bash", str(RUN_BOT)] + args
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            stdin=devnull, start_new_session=True)
    PID_PATH.write_text(str(proc.pid))
    print("Bot starting in the background.")
    print(f"  watch logs :  algobot logs")
    print(f"  stop       :  algobot stop")


def stop() -> None:
    pid = get_pid()
    if not pid or not is_running():
        print("Bot is not running.")
        return
    import signal
    import time
    # The bot runs as a supervisor process group (start_new_session=True), so
    # the group leader's pid == its pgid. Signal the whole group to take down
    # both the supervisor loop and the bot child it spawns.
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError as exc:
        print(f"Could not signal pid {pid}: {exc}")
        return
    print(f"Sent stop signal to bot (pid {pid}).")

    # The bot shuts down through scheduler.shutdown(wait=True), which waits for
    # the in-flight tick — a slow exchange call can hold that for a while, and a
    # tick that never returns leaves the python child alive after the supervisor
    # is gone. Escalate, or the "stopped" bot keeps burning CPU in the
    # background (and the next start then runs two bots on one database).
    deadline = time.time() + STOP_GRACE_SECONDS
    while time.time() < deadline:
        if not _group_alive(pid):
            print("Bot stopped.")
            return
        time.sleep(0.5)

    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError as exc:
        print(f"Could not kill pid {pid}: {exc}")
        return
    print(
        f"Bot did not exit within {STOP_GRACE_SECONDS}s "
        f"(a tick was still running) — killed (pid {pid})."
    )


def _group_alive(pgid: int) -> bool:
    """True while any process is left in the supervisor's process group.

    Checked with signal 0 on the *group*: the supervisor pid alone is not
    enough, because it can die while its python child lives on.
    """
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def status() -> None:
    from algotrading.config import load_settings
    s = load_settings()
    running = is_running()
    print(f"mode          : {s.mode}")
    print(f"running       : {'yes (pid ' + str(get_pid()) + ')' if running else 'no'}")
    print(f"telegram      : {'configured' if os.getenv('TELEGRAM_BOT_TOKEN') else 'missing'}")
    print(f"ai assistant  : {'enabled' if os.getenv('AI_API_KEY') else 'disabled'}")
    print(f"symbols       : {', '.join(s.market.symbols)}")


def show_logs(tail: int = 40) -> None:
    if not LOG_PATH.exists():
        print("No log file yet.")
        return
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    for line in lines[-tail:]:
        print(line)


# ---------------------------------------------------------------------------
# `algobot download` / `algobot export`
# ---------------------------------------------------------------------------

def _parse_flags(args: list[str]) -> dict[str, str]:
    """Parse ``--flag value`` / ``--flag=value`` into a dict.

    Argparse-free on purpose: this file's whole point is being runnable as a
    plain script with no surprises on Termux, and these two commands take a
    handful of optional flags. An unknown flag is reported by the caller, not
    silently ignored.
    """
    opts: dict[str, str] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            i += 1
            continue
        if "=" in token:
            key, _, value = token.partition("=")
            opts[key] = value
            i += 1
            continue
        following = args[i + 1] if i + 1 < len(args) else ""
        if following and not following.startswith("--"):
            opts[token] = following
            i += 2
        else:
            opts[token] = ""
            i += 1
    return opts


def _csv_list(value: str | None) -> list[str]:
    """Split a comma-separated flag value, dropping blanks."""
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _int_flag(opts: dict[str, str], name: str, default: int) -> int:
    raw = opts.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  {name} expects a whole number, got {raw!r} — using {default}.")
        return default


def download(args: list[str]) -> None:
    """`algobot download` — page history into the DB, optionally export it.

    The work happens in ``algotrading.market.download.download_history`` (the
    same ``backfill``/``upsert`` path the live tick uses), so this function only
    parses flags, prints progress and reports the result. The bot does not need
    to be running — but stop it first if it is, so the two are not writing to
    the same SQLite file at once.
    """
    if "-h" in args or "--help" in args:
        print(DOWNLOAD_HELP)
        return

    from algotrading.config import load_settings
    from algotrading.db import get_session, init_db
    from algotrading.market.binance_provider import BinanceMarketProvider
    from algotrading.market.candles import CandleStore
    from algotrading.market.download import (
        DEFAULT_MAX_ROWS,
        download_history,
        export_candles,
    )

    opts = _parse_flags(args)
    known = {"--symbols", "--intervals", "--days", "--max-rows", "--export", "--db"}
    unknown = sorted(set(opts) - known)
    if unknown:
        print(f"Unknown option(s): {', '.join(unknown)}")
        print(DOWNLOAD_HELP)
        return

    settings = load_settings()
    symbols = _csv_list(opts.get("--symbols")) or list(settings.market.symbols)
    intervals = _csv_list(opts.get("--intervals")) or list(settings.market.intervals)
    days = _int_flag(opts, "--days", int(settings.market.backfill_days))
    max_rows = _int_flag(opts, "--max-rows", DEFAULT_MAX_ROWS)
    export_path = opts.get("--export") or ""
    db_path = opts.get("--db") or settings.db_path

    print(
        f"Downloading {days} days of history for {', '.join(symbols)} "
        f"at {', '.join(intervals)} (row cap {max_rows})."
    )
    init_db(db_path)

    def _progress(symbol: str, interval: str, rows: int) -> None:
        print(f"  {symbol} {interval}: {rows} candles stored")

    with get_session(db_path) as session:
        store = CandleStore(session)
        report = download_history(
            BinanceMarketProvider(),
            store,
            symbols,
            intervals,
            days,
            max_rows=max_rows,
            progress=_progress,
        )

        if export_path:
            rows = []
            for entry in report["results"]:
                rows.extend(
                    store.get(entry["symbol"], entry["interval"], limit=EXPORT_LIMIT)
                )
            written = export_candles(rows, export_path)
            print(f"Exported {written} candles to {export_path}")

    resumed = sum(1 for r in report["results"] if r["resumed"])
    print(
        f"Done: {report['total_rows']} rows stored "
        f"across {len(report['results'])} symbol/interval pair(s) "
        f"({resumed} resumed)."
    )
    for entry in report["results"]:
        if entry["error"]:
            print(f"  !! {entry['symbol']} {entry['interval']}: {entry['error']}")
        elif entry["capped"]:
            print(
                f"  .. {entry['symbol']} {entry['interval']}: row cap reached — "
                "re-run to continue where this stopped"
            )
    if report["errors"]:
        print("Some pairs failed (see above); re-running resumes them.")


def export(args: list[str]) -> None:
    """`algobot export` — write trades/equity/summary files from the database.

    Read-only: every exporter reads the DB and writes one file. With no flags
    the help text is printed rather than an error, because that is what a
    non-technical user gets after trying the command bare.
    """
    if "-h" in args or "--help" in args:
        print(EXPORT_HELP)
        return

    from algotrading.config import load_settings
    from algotrading.db import get_session
    from algotrading.store.export import (
        export_equity_csv,
        export_summary_json,
        export_trades_csv,
        export_trades_json,
    )

    opts = _parse_flags(args)
    known = {
        "--dir", "--db", "--trades-csv", "--trades-json",
        "--equity-csv", "--summary-json",
    }
    unknown = sorted(set(opts) - known)
    if unknown:
        print(f"Unknown option(s): {', '.join(unknown)}")
        print(EXPORT_HELP)
        return

    settings = load_settings()
    db_path = opts.get("--db") or settings.db_path

    out_dir = opts.get("--dir") or ""
    targets = {
        "trades_csv": opts.get("--trades-csv") or (f"{out_dir}/trades.csv" if out_dir else ""),
        "trades_json": opts.get("--trades-json") or (f"{out_dir}/trades.json" if out_dir else ""),
        "equity_csv": opts.get("--equity-csv") or (f"{out_dir}/equity.csv" if out_dir else ""),
        "summary_json": opts.get("--summary-json") or (f"{out_dir}/summary.json" if out_dir else ""),
    }
    if not any(targets.values()):
        print(EXPORT_HELP)
        return

    with get_session(db_path) as session:
        if targets["trades_csv"]:
            n = export_trades_csv(session, targets["trades_csv"])
            print(f"trades      -> {targets['trades_csv']} ({n} rows)")
        if targets["trades_json"]:
            n = export_trades_json(session, targets["trades_json"])
            print(f"trades JSON -> {targets['trades_json']} ({n} rows)")
        if targets["equity_csv"]:
            n = export_equity_csv(
                session,
                targets["equity_csv"],
                initial_balance=float(settings.risk.paper_initial_balance),
            )
            print(f"equity      -> {targets['equity_csv']} ({n} points)")
        if targets["summary_json"]:
            summary = export_summary_json(
                session,
                targets["summary_json"],
                mode=settings.mode,
                initial_balance=float(settings.risk.paper_initial_balance),
            )
            counts = summary["counts"]
            print(
                f"summary     -> {targets['summary_json']} "
                f"({counts['trades']} trades, {counts['open_positions']} open)"
            )


def menu() -> None:
    while True:
        print()
        print("AlgoTrading — what do you want to do?")
        print("  1) Start the bot")
        print("  2) Stop the bot")
        print("  3) Status")
        print("  4) Show recent logs")
        print("  5) Run configuration setup")
        print("  6) Download market history")
        print("  7) Export trades / equity / summary")
        print("  0) Exit")
        choice = _input("  > ")
        if choice == "1":
            start([])
        elif choice == "2":
            stop()
        elif choice == "3":
            status()
        elif choice == "4":
            show_logs()
        elif choice == "5":
            run_setup()
        elif choice == "6":
            download([])
        elif choice == "7":
            export([])
        elif choice == "0":
            break
        else:
            print("  Invalid choice.")


def run_setup() -> None:
    """Interactive first-run configuration."""
    existing = read_env()
    updates = prompt_for_fields(existing)
    apply_updates(updates)

    # Validate the Telegram token if one was set/changed.
    token = updates.get("TELEGRAM_BOT_TOKEN", existing.get("TELEGRAM_BOT_TOKEN", ""))
    if token:
        if validate_telegram(token):
            print("Telegram token OK.")
        else:
            print("Could not verify the Telegram token (network or wrong token). "
                  "You can re-run setup later.")

    print()
    print("Setup complete. Start the bot with:   algobot")


def main() -> None:
    if sys.argv and sys.argv[0].endswith("algobot"):
        # Invoked as the `algobot` command
        cmd = sys.argv[1] if len(sys.argv) > 1 else "menu"
        if cmd == "start":
            start(sys.argv[2:])
        elif cmd == "stop":
            stop()
        elif cmd == "status":
            status()
        elif cmd in ("logs", "log"):
            tail = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 40
            show_logs(tail)
        elif cmd == "setup":
            run_setup()
        elif cmd == "download":
            download(sys.argv[2:])
        elif cmd == "export":
            export(sys.argv[2:])
        else:
            menu()
    else:
        # Invoked as the setup wizard (scripts/setup.py)
        run_setup()


if __name__ == "__main__":
    main()
