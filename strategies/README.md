# strategies/ — pluggable strategy files

Drop a `.py` file here and it becomes a usable strategy on the next reload
(the bot reloads this directory periodically; see `modules.strategy_plugin_paths`
and `reload_seconds` in `config/settings.yaml`).

Everything is validated before it runs. Plugins are pure signal producers —
they never place orders and cannot touch the network, the filesystem, or the
database. Execution stays deterministic in `algotrading/execution/`.

## File shape

```python
from algotrading.strategy import indicators as ta


class MyStrategy:
    """One-line description shown by the bot."""

    def __init__(self, params):
        self.params = params
        self.period = int(params.get("period", 14))

    def evaluate(self, symbol, candles):
        # candles: oldest -> newest, each with .open/.high/.low/.close/.volume
        if len(candles) < self.period + 1:
            return None
        closes = [c.close for c in candles]
        if closes[-1] > ta.sma(closes, self.period)[-1]:
            return Signal(
                strategy_id=0,          # the engine fills in the real id
                symbol=symbol,
                side="buy",             # "buy" opens, "sell" closes
                ref_price=closes[-1],
                rationale="close above SMA",
                risk={"position_pct": 0.2},
            )
        return None
```

`Signal` and `Candle` are provided automatically, as is `ta`
(`algotrading.strategy.indicators`) and `math`.

## Optional metadata

Appended by the loader, or declare it yourself:

```python
STRATEGY = MyStrategy
STRATEGY_DESCRIPTION = "Close above SMA"
STRATEGY_PARAMS = [{"name": "period", "type": "int", "default": 14, "min": 2, "max": 200}]
STRATEGY_INDICATORS = ["sma"]
```

## Rules

- The class must define `evaluate(self, symbol, candles)`.
- Imports of `os`, `sys`, `socket`, `subprocess`, `requests`, ... are rejected,
  as are `eval` / `exec` / `open` / `__import__`.
- Names of built-in strategies (`ema_crossover`, `ensemble`, ...) cannot be
  reused here. Edit built-ins through a parameter change instead.
- A new strategy only goes live after a human approves the AI's proposal
  (Telegram ✅). The assistant can write these files; it cannot activate them.

## AI-authored strategies

The AI creates/edits files here through the normal approval flow
(`new_strategy` / `edit_strategy` proposals). The resulting file is a normal
plugin you can hand-edit afterwards.
