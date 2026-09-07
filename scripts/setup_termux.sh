#!/usr/bin/env bash
# AlgoTrading Termux setup: packages, venv, deps, .env, DB init, wake-lock.
#
# Usage:  bash scripts/setup_termux.sh
#
# Optional extras (read before running):
#   - termux-services (runit) auto-restart: after setup, copy the bot service
#     into ~/.termux/services/algotrading/run  (see README "Running as a service").
#   - This script never touches files outside the project + Termux basics.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "==> [1/6] Installing system packages (Termux)"
pkg install -y python python-pip sqlite openssl termux-api || {
    echo "pkg install failed — run 'pkg update' first and retry." >&2
    exit 1
}

echo "==> [2/6] Creating Python virtualenv (.venv)"
if [ ! -d .venv ]; then
    python -m venv .venv
fi

echo "==> [3/6] Installing Python dependencies"
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "==> [4/6] Creating .env from template (if missing)"
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env — edit it and fill in TELEGRAM_BOT_TOKEN (+ API keys for live/AI)."
else
    echo ".env already exists; leaving it untouched."
fi

echo "==> [5/6] Initializing the database"
.venv/bin/python scripts/init_db.py

echo "==> [6/6] Acquiring wake-lock (keeps CPU awake while trading)"
if command -v termux-wake-lock >/dev/null 2>&1; then
    termux-wake-lock && echo "wake-lock held"
else
    echo "termux-wake-lock not found — install termux-api (already requested above)."
fi

echo
echo "Setup complete. Next steps:"
echo "  1. Edit .env  (TELEGRAM_BOT_TOKEN at minimum)"
echo "  2. Review config/settings.yaml (mode: paper by default)"
echo "  3. Run:  bash scripts/run_bot.sh"
echo "     or:   .venv/bin/python -m algotrading.main --demo-data"