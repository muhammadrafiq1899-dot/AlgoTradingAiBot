"""Shadow backtesting: replay stored candles through a strategy.

`runner` is the replay itself, `metrics` the statistics layer on top of it,
`stored` the DB adapter that feeds both. The package stays free of execution,
telegram and scheduler imports on purpose: a backtest must never be able to
place an order or block the bot's tick thread.
"""
from algotrading.backtest.runner import (
    BacktestResult,
    BacktestTrade,
    CandleWindow,
    run_backtest,
)

__all__ = ["BacktestResult", "BacktestTrade", "CandleWindow", "run_backtest"]
