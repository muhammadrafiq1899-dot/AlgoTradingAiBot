#!/usr/bin/env bash
# AlgoTrading supervisor loop: run the bot, restart on crash with backoff.
#
# Usage:  bash scripts/run_bot.sh [--demo-data] [--no-telegram]
#
# Passes extra args through to algotrading.main. Heartbeat: the bot touches
# data/heartbeat every 60s; if this loop finds the process alive but the
# heartbeat stale, it kills and restarts it. Restart delay backs off when
# runs are short (crash loop) and resets after a healthy run.
set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON=.venv/bin/python
HEARTBEAT=data/heartbeat
MAX_STALE_SECONDS=300
MAX_RESTART_DELAY=60
RESTART_DELAY=5
LOCKFILE=data/run_bot.lock
DB_FILE=data/algotrading.db
BACKUP_DIR=data/backups
BACKUP_RETENTION_DAYS=7

if [ ! -x "$PYTHON" ]; then
    echo "No .venv found — run: bash scripts/setup_termux.sh" >&2
    exit 1
fi

# Single-instance guard: only one supervisor loop may run. A second start
# (e.g. algobot start + a manual bash scripts/run_bot.sh) would create two
# polling bots fighting over the same token and heartbeat file.
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "run_bot.sh is already running (pid $(cat "$LOCKFILE" 2>/dev/null)) — not starting a second supervisor." >&2
    exit 1
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

# A heartbeat left over from a previous session would make the watchdog think
# the freshly-started bot is stale and kill it before it can write its first
# beat (first tick is ~60s). Clear it now so a new boot starts clean.
rm -f "$HEARTBEAT"

# Ensure backup directory exists
mkdir -p "$BACKUP_DIR"

# Cleanup old backups (older than BACKUP_RETENTION_DAYS)
find "$BACKUP_DIR" -type f -name "algotrading-*.db" -mtime +"$BACKUP_RETENTION_DAYS" -delete 2>/dev/null || true

# Create SQLite backup before each start (point-in-time recovery)
if [ -f "$DB_FILE" ]; then
    BACKUP_NAME="algotrading-$(date -u +%F-%H%M%S).db"
    if "$PYTHON" -c "import sqlite3; sqlite3.connect('$DB_FILE').backup(sqlite3.connect('$BACKUP_DIR/$BACKUP_NAME'))" 2>/dev/null; then
        echo "[$(date -u +%H:%M:%S)] DB backed up to $BACKUP_DIR/$BACKUP_NAME"
    else
        echo "[$(date -u +%H:%M:%S)] WARNING: DB backup failed (continuing anyway)" >&2
    fi
fi

while true; do
    START=$(date +%s)
    "$PYTHON" -m algotrading.main "$@" &
    BOT_PID=$!
    echo "[$(date -u +%H:%M:%S)] bot started (pid $BOT_PID); args: $*"

    # Watchdog: wait for the bot to exit, but also restart it if the
    # heartbeat goes stale while the process is still alive.
    # Grace period: never kill before MAX_STALE_SECONDS of uptime — a slow
    # boot (e.g. wake-lock or network stall) must not be killed before it
    # writes its first heartbeat.
    while kill -0 "$BOT_PID" 2>/dev/null; do
        UPTIME=$(( $(date +%s) - START ))
        if [ -f "$HEARTBEAT" ] && [ "$UPTIME" -gt "$MAX_STALE_SECONDS" ]; then
            AGE=$(( $(date +%s) - $(stat -c %Y "$HEARTBEAT") ))
            if [ "$AGE" -gt "$MAX_STALE_SECONDS" ]; then
                echo "[$(date -u +%H:%M:%S)] heartbeat stale (${AGE}s) — killing bot" >&2
                kill "$BOT_PID" 2>/dev/null
            fi
        fi
        sleep 10
    done

    wait "$BOT_PID" 2>/dev/null
    EXIT_CODE=$?
    UPTIME=$(( $(date +%s) - START ))
    echo "[$(date -u +%H:%M:%S)] bot exited (code $EXIT_CODE) after ${UPTIME}s"

    # Backoff: short runs mean a crash loop; healthy runs reset the delay.
    if [ "$UPTIME" -lt 60 ]; then
        RESTART_DELAY=$(( RESTART_DELAY * 2 ))
        if [ "$RESTART_DELAY" -gt "$MAX_RESTART_DELAY" ]; then
            RESTART_DELAY=$MAX_RESTART_DELAY
        fi
    else
        RESTART_DELAY=5
    fi
    echo "[$(date -u +%H:%M:%S)] restarting in ${RESTART_DELAY}s"
    sleep "$RESTART_DELAY"
done