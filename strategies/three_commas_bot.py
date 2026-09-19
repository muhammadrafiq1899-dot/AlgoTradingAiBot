"""3Commas Bot (Pine v5, "Bj Bot") converted to an AlgoTrading strategy plugin.

THE ORIGINAL
    long entry   : ta.crossover(ma1, ma2)          (MA1 21, MA2 50, selectable type)
    short entry  : ta.crossunder(ma1, ma2)         (mirror image)
    initial stop : swing low  - ATR * RiskAdjustment   (short: swing high + ATR * RiskM)
    target       : entry + Reward:Risk * risk distance ("Use Limit exit")
    trailing     : ATR trailing stop, armed once price reaches
                   rrExit * distance-to-target (0 = armed immediately)
    filters      : ignore-trades session window, start/end date window
    risk         : strategy.risk.max_drawdown(percent_of_equity)

WHAT THIS PORT KEEPS
    * the MA-pair crossover entry, with the same MA menu: EMA, HEMA (Heikin-Ashi
      EMA), SMA, HMA, WMA, VWMA, VWAP, DEMA, T3
    * swing-based initial stop (RiskM * ATR), Reward:Risk target, and the ATR
      trailing stop with its High/Low | Close | Open source and rrExit arming level
    * the ignore-session filter and the start/end date filter
    * one signal per bar, driven by the same conditions as the Pine version

DELIBERATE DIFFERENCES (read these before trusting it live)
    1. LONG-ONLY. The bot trades Binance spot: it can buy and later sell what it
       owns, but it cannot short. So "shortTrades" and "FLIP" from the original
       have no equivalent. Instead, the bearish MA cross (`ma1 crossunder ma2`)
       is offered as an exit (`exit_on_cross`), which is the long-only reading of
       the same event.
    2. STOPS AND TARGETS ARE SIGNALS, NOT BROKER ORDERS. The bot's execution is
       deterministic: a strategy emits buy/sell and the engine sends a MARKET
       order on the next tick. A Pine `strategy.exit(stop=..., limit=...)` would
       be a resting order on the exchange; here the level is checked against the
       bar's high/low and the exit is a market signal, so fills are a tick later
       and can slip. Set the exchange-side protection you want in
       `config/settings.yaml` (`risk.trailing_stop_pct`, max positions) as well.
    3. MAX DRAWDOWN IS THE BOT'S JOB. `strategy.risk.max_drawdown` is replaced by
       the bot's own risk layer (`risk.max_daily_loss_pct`, exposure caps) — a
       pure strategy has no equity to measure.
    4. THE BOT REBUILDS THE STRATEGY EVERY TICK, so there is no instance memory.
       Each `evaluate()` replays the window it was handed to rebuild any trade in
       progress (entry price, stop, target, trail). That is why every threshold
       lives in `_replay` instead of `self`. Consequence: a trade opened before
       the window starts is invisible — the engine's snapshot holds
       SNAPSHOT_LIMIT candles (~12 days on 1h). The `exit_on_cross` exit and the
       engine's own risk rules still protect that position.
    5. Session filter takes plain hour inputs (0-23, with a UTC offset) instead of
       Pine's session strings; default 00:00-03:00 at GMT-6 matches the original.
"""

# --- metadata (the loader reads these when present) --------------------------

MA_TYPES = ["EMA", "HEMA", "SMA", "HMA", "WMA", "DEMA", "VWMA", "VWAP", "T3"]

STRATEGY_DESCRIPTION = (
    "3Commas Bot conversion (long-only): MA cross entry with ATR swing stop, "
    "Reward:Risk target and ATR trailing exit, plus session and date filters"
)

STRATEGY_INDICATORS = ["ema", "sma", "atr"]

STRATEGY_PARAMS = [
    {"name": "long_trades", "type": "bool", "default": True},
    {"name": "exit_on_cross", "type": "bool", "default": True},
    {"name": "use_limit", "type": "bool", "default": True},
    {"name": "trail_stop", "type": "bool", "default": False},
    {"name": "rnr", "type": "float", "default": 1.0, "min": 0.0, "max": 20.0},
    {"name": "risk_m", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0},
    {"name": "swing_lookback", "type": "int", "default": 5, "min": 1, "max": 100},
    {"name": "atr_len", "type": "int", "default": 14, "min": 1, "max": 100},
    {"name": "trail_stop_size", "type": "float", "default": 1.0, "min": 0.0, "max": 20.0},
    {"name": "trail_source", "type": "str", "default": "High/Low",
     "enum": ["High/Low", "Close", "Open"]},
    {"name": "rr_exit", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
    {"name": "ma_type_1", "type": "str", "default": "EMA", "enum": MA_TYPES},
    {"name": "ma_type_2", "type": "str", "default": "EMA", "enum": MA_TYPES},
    {"name": "ma_length_1", "type": "int", "default": 21, "min": 1, "max": 500},
    {"name": "ma_length_2", "type": "int", "default": 50, "min": 1, "max": 500},
    {"name": "ignore_session", "type": "bool", "default": False},
    {"name": "session_start_hour", "type": "int", "default": 0, "min": 0, "max": 23},
    {"name": "session_end_hour", "type": "int", "default": 3, "min": 0, "max": 23},
    {"name": "session_tz_offset", "type": "int", "default": -6, "min": -12, "max": 14},
    {"name": "start_ts", "type": "int", "default": 0, "min": 0},
    {"name": "end_ts", "type": "int", "default": 0, "min": 0},
    {"name": "position_pct", "type": "float", "default": 0.2, "min": 0.01, "max": 0.5},
]


# --- moving averages (only ta.sma / ta.ema / ta.vwap_anchored are injected) ---

def _last(values, default=None):
    """Most recent value, or `default` when the series is short/NaN."""
    for value in reversed(list(values)):
        if value is not None and value == value:      # NaN != NaN
            return value
    return default


def _wma(values, period):
    """Linearly weighted MA (Pine's ta.wma)."""
    n = len(values)
    out = [float("nan")] * n
    if period < 1:
        return out
    denom = period * (period + 1) / 2.0
    for i in range(period - 1, n):
        window = values[i - period + 1:i + 1]
        if any(v is None or v != v for v in window):
            continue
        total = 0.0
        for j, value in enumerate(window):
            total += value * (j + 1)
        out[i] = total / denom
    return out


def _hma(values, period):
    """Hull MA: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    n = len(values)
    out = [float("nan")] * n
    half = max(1, int(period / 2))
    root = max(1, int(period ** 0.5))
    fast, slow = _wma(values, half), _wma(values, period)
    diff = [
        (2 * fast[i] - slow[i])
        if fast[i] == fast[i] and slow[i] == slow[i] and fast[i] is not None and slow[i] is not None
        else float("nan")
        for i in range(n)
    ]
    return _wma(diff, root)


def _vwma(values, volumes, period):
    """Volume weighted MA over a rolling window."""
    n = len(values)
    out = [float("nan")] * n
    for i in range(period - 1, n):
        pv = 0.0
        vol = 0.0
        for j in range(i - period + 1, i + 1):
            pv += values[j] * volumes[j]
            vol += volumes[j]
        out[i] = pv / vol if vol > 0 else float("nan")
    return out


def _dema(values, period):
    first = ta.ema(values, period)
    # ta.ema skips NaN and propagates the last value, so feed it straight through.
    second = ta.ema(first, period)
    return [
        (2 * first[i] - second[i]) if first[i] == first[i] and second[i] == second[i] else float("nan")
        for i in range(len(values))
    ]


def _t3(values, period, volume_factor=0.7):
    """Tillson T3, matching the Pine math (six chained EMAs)."""
    e1 = ta.ema(values, period)
    e2 = ta.ema(e1, period)
    e3 = ta.ema(e2, period)
    e4 = ta.ema(e3, period)
    e5 = ta.ema(e4, period)
    e6 = ta.ema(e5, period)
    ab = volume_factor
    c1 = -ab * ab * ab
    c2 = 3 * ab * ab + 3 * ab * ab * ab
    c3 = -6 * ab * ab - 3 * ab - 3 * ab * ab * ab
    c4 = 1 + 3 * ab + ab * ab * ab + 3 * ab * ab
    out = []
    for i in range(len(values)):
        parts = (e6[i], e5[i], e4[i], e3[i])
        if any(p != p for p in parts):
            out.append(float("nan"))
        else:
            out.append(c1 * e6[i] + c2 * e5[i] + c3 * e4[i] + c4 * e3[i])
    return out


def _heikin_ashi_open(candles):
    """Pine's `_haOpen()` series (used by the HEMA option)."""
    out = []
    previous_open = None
    previous_close = None
    for candle in candles:
        ha_close = (candle.open + candle.high + candle.low + candle.close) / 4.0
        if previous_open is None or previous_close is None:
            ha_open = (candle.open + candle.close) / 2.0
        else:
            ha_open = (previous_open + previous_close) / 2.0
        out.append(ha_open)
        previous_open, previous_close = ha_open, ha_close
    return out


def _session_vwap(candles):
    """VWAP anchored to the start of the last UTC day in the window."""
    highs, lows, closes_, volumes = ta.highs(candles), ta.lows(candles), ta.closes(candles), [
        c.volume for c in candles
    ]
    anchor = 0
    day_start = candles[-1].ts // 86_400_000
    for i in range(len(candles) - 1, -1, -1):
        if candles[i].ts // 86_400_000 != day_start:
            anchor = i + 1
            break
    return ta.vwap_anchored(highs, lows, closes_, volumes, anchor)


def _build_ma(kind, candles, period):
    """Dispatch on the MA menu from the Pine inputs."""
    closes_ = ta.closes(candles)
    if kind == "SMA":
        return ta.sma(closes_, period)
    if kind == "EMA":
        return ta.ema(closes_, period)
    if kind == "WMA":
        return _wma(closes_, period)
    if kind == "HMA":
        return _hma(closes_, period)
    if kind == "HEMA":
        return ta.ema(_heikin_ashi_open(candles), period)
    if kind == "VWMA":
        return _vwma(closes_, [c.volume for c in candles], period)
    if kind == "VWAP":
        return _session_vwap(candles)
    if kind == "DEMA":
        return _dema(closes_, period)
    if kind == "T3":
        return _t3(closes_, period)
    return ta.ema(closes_, period)      # unknown type: behave like the default


class ThreeCommasBot:
    """Long-only port of the Pine "3Commas Bot": MA cross + ATR risk management."""

    name = "three_commas_bot"

    def __init__(self, params=None):
        params = params or {}
        self.params = params
        self.long_trades = bool(params.get("long_trades", True))
        self.exit_on_cross = bool(params.get("exit_on_cross", True))
        self.use_limit = bool(params.get("use_limit", True))
        self.trail_stop = bool(params.get("trail_stop", False))
        self.rnr = float(params.get("rnr", 1.0))
        self.risk_m = float(params.get("risk_m", 1.0))
        self.swing_lookback = max(1, int(params.get("swing_lookback", 5)))
        self.atr_len = max(1, int(params.get("atr_len", 14)))
        self.trail_stop_size = float(params.get("trail_stop_size", 1.0))
        self.trail_source = str(params.get("trail_source", "High/Low"))
        self.rr_exit = float(params.get("rr_exit", 0.0))
        self.ma_type_1 = str(params.get("ma_type_1", "EMA"))
        self.ma_type_2 = str(params.get("ma_type_2", "EMA"))
        self.ma_length_1 = max(1, int(params.get("ma_length_1", 21)))
        self.ma_length_2 = max(1, int(params.get("ma_length_2", 50)))
        self.ignore_session = bool(params.get("ignore_session", False))
        self.session_start_hour = int(params.get("session_start_hour", 0)) % 24
        self.session_end_hour = int(params.get("session_end_hour", 3)) % 24
        self.session_tz_offset = int(params.get("session_tz_offset", -6))
        self.start_ts = int(params.get("start_ts", 0) or 0)
        self.end_ts = int(params.get("end_ts", 0) or 0)
        self.position_pct = float(params.get("position_pct", 0.2))

    # --- filters --------------------------------------------------------------

    def _in_ignore_session(self, ts_ms):
        """Pine's `useTimeFilter` window: True when trading must pause."""
        if not self.ignore_session:
            return False
        hour = int((ts_ms // 3_600_000 + self.session_tz_offset) % 24)
        start, end = self.session_start_hour, self.session_end_hour
        if start == end:
            return False
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end      # window wraps midnight

    def _in_date_window(self, ts_ms):
        if self.start_ts and ts_ms < self.start_ts:
            return False
        if self.end_ts and ts_ms > self.end_ts:
            return False
        return True

    def _can_enter(self, ts_ms):
        return self._in_date_window(ts_ms) and not self._in_ignore_session(ts_ms)

    # --- trade reconstruction -------------------------------------------------

    def _replay(self, candles, ma1, ma2, atr_series):
        """Rebuild the trade in progress by walking the window.

        Returns ``(state, exit_on_last_bar, entry_on_last_bar)`` where state is
        None when flat, else a dict with the live stop/target/trail. The last-bar
        flags are what the caller turns into a signal: the bar where the replay
        opens or closes a trade is exactly the bar the bot must act on.

        Stateless by design: the engine builds a fresh instance every tick, so
        nothing may live on `self`.
        """
        n = len(candles)
        highs, lows, closes_, opens = (
            ta.highs(candles), ta.lows(candles), ta.closes(candles),
            [c.open for c in candles],
        )
        state = None
        exit_now = False
        entry_now = False

        for i in range(n):
            last_bar = i == n - 1
            a = atr_series[i]
            atr_ok = a is not None and a == a
            ma_ok = bool(
                i >= 1
                and ma1[i] == ma1[i] and ma2[i] == ma2[i]
                and ma1[i - 1] == ma1[i - 1] and ma2[i - 1] == ma2[i - 1]
            )
            cross_up = ma_ok and ma1[i - 1] <= ma2[i - 1] and ma1[i] > ma2[i]

            if state is not None:
                stop = state["trail"] if self.trail_stop else state["stop"]

                # Target hit inside the bar (Pine fills the limit order intrabar).
                if self.use_limit and state["target"] is not None and highs[i] >= state["target"]:
                    state = None
                    exit_now, entry_now = last_bar, False
                    continue
                # Stop touched inside the bar.
                if lows[i] <= stop:
                    state = None
                    exit_now, entry_now = last_bar, False
                    continue
                # Long-only reading of the original short entry.
                if self.exit_on_cross and ma_ok and ma1[i - 1] >= ma2[i - 1] and ma1[i] < ma2[i]:
                    state = None
                    exit_now, entry_now = last_bar, False
                    continue
                if last_bar:
                    exit_now, entry_now = False, False

                # Arm the trail once price reaches rr_exit of the way to target.
                if self.trail_stop and atr_ok:
                    if self.rr_exit > 0.0:
                        if self.use_limit and state["target"] is not None:
                            trigger = state["entry"] + (state["target"] - state["entry"]) * self.rr_exit
                            if highs[i] >= trigger:
                                state["armed"] = True
                    else:
                        state["armed"] = True

                    if state["armed"]:
                        if self.trail_source == "Close":
                            src = closes_[i - 1] if i >= 1 else closes_[i]
                        elif self.trail_source == "Open":
                            src = opens[i - 1] if i >= 1 else opens[i]
                        else:
                            look = lows[max(0, i - self.swing_lookback + 1):i + 1]
                            src = min(look)
                        candidate = src - a * self.trail_stop_size
                        if candidate > state["trail"]:
                            state["trail"] = candidate
                continue

            # Flat: look for an entry.
            exit_now, entry_now = False, False
            if not (self.long_trades and cross_up and atr_ok and self._can_enter(candles[i].ts)):
                continue
            entry = closes_[i]
            look = lows[max(0, i - self.swing_lookback + 1):i + 1]
            stop = min(look) - a * self.risk_m
            risk = entry - stop
            target = entry + self.rnr * risk if (self.use_limit and risk > 0) else None
            state = {
                "entry": entry,
                "stop": stop,
                "target": target,
                "trail": stop,
                "armed": self.rr_exit == 0.0,
            }
            if last_bar:
                entry_now = True

        return state, exit_now, entry_now

    # --- signal ---------------------------------------------------------------

    def evaluate(self, symbol, candles):
        if len(candles) < max(self.ma_length_1, self.ma_length_2, self.atr_len) + 2:
            return None

        ma1 = _build_ma(self.ma_type_1, candles, self.ma_length_1)
        ma2 = _build_ma(self.ma_type_2, candles, self.ma_length_2)
        atr_series = ta.atr(ta.highs(candles), ta.lows(candles), ta.closes(candles), self.atr_len)
        if _last(atr_series) is None:
            return None

        state, exit_now, entry_now = self._replay(candles, ma1, ma2, atr_series)
        price = candles[-1].close

        if exit_now:
            return Signal(
                strategy_id=0, symbol=symbol, side="sell", ref_price=price,
                rationale="3Commas exit: stop/target/MA cross triggered",
                risk={"position_pct": self.position_pct},
                params=dict(self.params),
            )

        if entry_now:
            atr_last = _last(atr_series)
            stop = min(ta.lows(candles)[-self.swing_lookback:]) - atr_last * self.risk_m
            return Signal(
                strategy_id=0, symbol=symbol, side="buy", ref_price=price,
                rationale=(
                    f"3Commas entry: {self.ma_type_1}({self.ma_length_1}) crossed above "
                    f"{self.ma_type_2}({self.ma_length_2}), stop {stop:.4f}"
                ),
                risk={"position_pct": self.position_pct},
                params=dict(self.params),
            )

        return None
