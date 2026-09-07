"""Unit tests for pure-Python indicators against known values."""
import math

from algotrading.strategy import indicators as ta


def test_sma_known():
    out = ta.sma([1, 2, 3, 4, 5], 3)
    assert math.isnan(out[0]) and math.isnan(out[1])
    assert out[2:] == [2.0, 3.0, 4.0]


def test_sma_single():
    out = ta.sma([10, 20], 1)
    assert out == [10.0, 20.0]


def test_ema_known():
    # alpha = 2/(3+1) = 0.5; seed = SMA(1,2,3)=2
    out = ta.ema([1, 2, 3, 4, 5], 3)
    assert math.isnan(out[0]) and math.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[3] - 3.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def test_rsi_all_gains_is_100():
    out = ta.rsi([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], 14)
    assert out[-1] == 100.0


def test_rsi_wilder_seed():
    vals = [
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
        45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28,
    ]
    out = ta.rsi(vals, 14)
    # First RSI value at index 14; independently computed ~70.46
    assert abs(out[14] - 70.46) < 1.0


def test_atr_basic():
    # Flat ranges: each candle spans high=10, low=9 -> TR = 1 -> ATR = 1
    highs = [10.0] * 20
    lows = [9.0] * 20
    closes = [9.5] * 20
    out = ta.atr(highs, lows, closes, 14)
    assert math.isnan(out[14]) is False
    assert out[-1] == 1.0


def test_last_valid():
    assert ta.last_valid([float("nan"), float("nan"), 3.0, float("nan")]) == 3.0
    assert ta.last_valid([float("nan"), float("nan")]) is None
