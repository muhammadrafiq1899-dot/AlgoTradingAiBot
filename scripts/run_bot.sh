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

if [ ! -x "$PYTHON" ]; then
    echo "No .venv found — run: bash scripts/setup_termux.sh" >&2
    exit 1
fi

while true; do
    START=$(date +%s)
    "$PYTHON" -m algotrading.main "$@" &
    BOT_PID=$!
    echo "[$(date -u +%H:%M:%S)] bot started (pid $BOT_PID); args: $*"

    # Watchdog: wait for the bot to exit, but also restart it if the
    # heartbeat goes stale while the process is still alive.
    while kill -0 "$BOT_PID" 2>/dev/null; do
        if [ -f "$HEARTBEAT" ]; then
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