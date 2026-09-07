#!/usr/bin/env python3
"""Interactive first-run setup for AlgoTrading.

Guides a non-technical user through configuring the Telegram bot connection
and (optionally) the AI API key. Writes secrets into .env and initializes the
database.

Usage:  python scripts/setup.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algotrading.cli_setup import run_setup  # noqa: E402


if __name__ == "__main__":
    run_setup()
