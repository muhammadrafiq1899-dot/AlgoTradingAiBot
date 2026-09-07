#!/usr/bin/env python3
"""Initialize the AlgoTrading database and seed starter strategies.

Usage:  python scripts/init_db.py
Creates data/algotrading.db, tables, and seeds one active starter strategy.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root or from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from algotrading.config import load_settings  # noqa: E402
from algotrading.db import get_session, init_db  # noqa: E402
from algotrading.db.seed import ensure_seeded  # noqa: E402


def main() -> None:
    settings = load_settings()
    init_db(settings.db_path)
    print(f"Database initialized at {settings.db_path} (schema v1).")
    with get_session(settings.db_path) as session:
        n = ensure_seeded(session)
    if n:
        print(f"Seeded {n} strategies from config/strategies.yaml.")
    else:
        print("Strategies already seeded; nothing to do.")


if __name__ == "__main__":
    main()