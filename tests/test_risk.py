"""Unit tests for the risk module: sizing, exposure caps, cooldowns."""
from algotrading.config import RiskConfig
from algotrading.execution.risk import RiskManager


def _cfg(**overrides) -> RiskConfig:
    base = dict(
        risk_per_trade_pct=1.0,
        max_position_pct=20.0,
        max_open_positions=3,
        cooldown_seconds=300,
        slippage_pct=0.05,
    )
    base.update(overrides)
    return RiskConfig(**base)


def test_size_with_atr_risks_only_budget():
    rm = RiskManager(_cfg(risk_per_trade_pct=1.0))
    qty = rm.size_position(balance=100_000, price=50_000, atr_value=5_000)
    # risk budget = 1000; stop distance = 5000 -> qty = 0.2
    # notional 0.2*50k = 10k stays under the 20% cap -> budget is the binder
    assert abs(qty - 0.2) < 1e-6


def test_size_capped_by_max_position_pct():
    rm = RiskManager(_cfg(risk_per_trade_pct=1.0, max_position_pct=10.0))
    # atr tiny -> huge qty, but cap: 10% of 100k = 10k notional / 50k = 0.2
    qty = rm.size_position(balance=100_000, price=50_000, atr_value=1)
    assert abs(qty - 0.2) < 1e-6


def test_size_without_atr_uses_fraction():
    rm = RiskManager(_cfg(risk_per_trade_pct=2.0))
    qty = rm.size_position(balance=100_000, price=100, atr_value=None)
    # 2% of balance / price
    assert abs(qty - 20.0) < 1e-6


def test_max_open_positions_guard():
    rm = RiskManager(_cfg(max_open_positions=2))
    assert rm.check_buy(symbol="X", balance=1000, open_positions=1, last_entry_ts=None).allowed
    assert not rm.check_buy(symbol="X", balance=1000, open_positions=2, last_entry_ts=None).allowed


def test_cooldown_guard():
    import time
    rm = RiskManager(_cfg(cooldown_seconds=300))
    recent = time.time() * 1000  # just now -> cooldown active
    assert not rm.check_buy(symbol="X", balance=1000, open_positions=0, last_entry_ts=recent).allowed
    stale = recent - 400_000  # > 300s ago
    assert rm.check_buy(symbol="X", balance=1000, open_positions=0, last_entry_ts=stale).allowed


def test_sell_requires_position():
    rm = RiskManager(_cfg())
    assert not rm.check_sell(has_position=False).allowed
    assert rm.check_sell(has_position=True).allowed
