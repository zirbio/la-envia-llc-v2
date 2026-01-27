"""
OHLCV DataFrame generators for testing.

Provides functions to create realistic market data scenarios:
- Trending up/down
- Breakout patterns
- Volume spikes
- MACD crossovers
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_ohlcv_df(
    n_bars: int = 50,
    base_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0,
    base_volume: int = 1_000_000,
    volume_volatility: float = 0.3,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create a realistic OHLCV DataFrame.

    Args:
        n_bars: Number of bars
        base_price: Starting price
        volatility: Daily volatility (as decimal, e.g., 0.02 = 2%)
        trend: Trend direction (-1 to 1, 0 = no trend)
        base_volume: Average volume
        volume_volatility: Volume variation (as decimal)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with open, high, low, close, volume columns
    """
    np.random.seed(seed)

    # Generate returns with trend
    returns = np.random.randn(n_bars) * volatility + (trend * volatility / 2)
    close_prices = base_price * np.cumprod(1 + returns)

    # Generate OHLC from close
    high_prices = close_prices * (1 + np.abs(np.random.randn(n_bars) * volatility / 2))
    low_prices = close_prices * (1 - np.abs(np.random.randn(n_bars) * volatility / 2))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    # Ensure OHLC relationships
    high_prices = np.maximum(high_prices, np.maximum(close_prices, open_prices))
    low_prices = np.minimum(low_prices, np.minimum(close_prices, open_prices))

    # Generate volume
    volumes = (base_volume * (1 + np.random.randn(n_bars) * volume_volatility)).astype(int)
    volumes = np.maximum(volumes, base_volume // 10)  # Minimum volume

    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)


def create_trending_up_df(
    n_bars: int = 50,
    base_price: float = 100.0,
    trend_strength: float = 0.5,
    seed: int = 42
) -> pd.DataFrame:
    """Create DataFrame with clear uptrend."""
    return create_ohlcv_df(
        n_bars=n_bars,
        base_price=base_price,
        volatility=0.01,
        trend=trend_strength,
        seed=seed
    )


def create_trending_down_df(
    n_bars: int = 50,
    base_price: float = 100.0,
    trend_strength: float = 0.5,
    seed: int = 42
) -> pd.DataFrame:
    """Create DataFrame with clear downtrend."""
    return create_ohlcv_df(
        n_bars=n_bars,
        base_price=base_price,
        volatility=0.01,
        trend=-trend_strength,
        seed=seed
    )


def create_breakout_long_df(
    n_bars: int = 50,
    base_price: float = 100.0,
    range_high: float = 100.50,
    breakout_bar: int = 40,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create DataFrame simulating a LONG breakout.

    First bars consolidate below range_high, then break above.
    """
    np.random.seed(seed)

    # Consolidation phase
    consol_bars = breakout_bar
    consol_prices = base_price + np.random.randn(consol_bars) * 0.2
    consol_prices = np.clip(consol_prices, base_price - 0.5, range_high - 0.1)

    # Breakout phase
    break_bars = n_bars - breakout_bar
    break_prices = np.linspace(range_high, range_high + 1.0, break_bars)
    break_prices += np.random.randn(break_bars) * 0.1

    prices = np.concatenate([consol_prices, break_prices])

    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

    return pd.DataFrame({
        'open': prices - 0.1,
        'high': prices + np.abs(np.random.randn(n_bars) * 0.2),
        'low': prices - np.abs(np.random.randn(n_bars) * 0.2),
        'close': prices,
        'volume': [1_500_000] * consol_bars + [3_000_000] * break_bars  # Volume spike on breakout
    }, index=dates)


def create_breakout_short_df(
    n_bars: int = 50,
    base_price: float = 100.0,
    range_low: float = 98.0,
    breakout_bar: int = 40,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create DataFrame simulating a SHORT breakout.

    First bars consolidate above range_low, then break below.
    """
    np.random.seed(seed)

    # Consolidation phase
    consol_bars = breakout_bar
    consol_prices = base_price + np.random.randn(consol_bars) * 0.2
    consol_prices = np.clip(consol_prices, range_low + 0.1, base_price + 0.5)

    # Breakdown phase
    break_bars = n_bars - breakout_bar
    break_prices = np.linspace(range_low, range_low - 1.0, break_bars)
    break_prices += np.random.randn(break_bars) * 0.1

    prices = np.concatenate([consol_prices, break_prices])

    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

    return pd.DataFrame({
        'open': prices + 0.1,
        'high': prices + np.abs(np.random.randn(n_bars) * 0.2),
        'low': prices - np.abs(np.random.randn(n_bars) * 0.2),
        'close': prices,
        'volume': [1_500_000] * consol_bars + [3_000_000] * break_bars
    }, index=dates)


def create_volume_spike_df(
    n_bars: int = 50,
    base_price: float = 100.0,
    spike_bar: int = -1,
    spike_multiplier: float = 5.0,
    base_volume: int = 500_000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create DataFrame with a volume spike.

    Args:
        spike_bar: Bar index for spike (-1 = last bar)
        spike_multiplier: Volume multiplier for spike
    """
    np.random.seed(seed)

    prices = np.linspace(base_price, base_price + 5, n_bars)
    volumes = [base_volume] * n_bars

    if spike_bar == -1:
        spike_bar = n_bars - 1
    volumes[spike_bar] = int(base_volume * spike_multiplier)

    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

    return pd.DataFrame({
        'open': prices - 0.5,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices,
        'volume': volumes
    }, index=dates)


def create_macd_bullish_crossover_df(
    n_bars: int = 60,
    base_price: float = 100.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create DataFrame that produces a MACD bullish crossover.

    Price pattern: decline followed by recovery.
    """
    np.random.seed(seed)

    # Decline phase (histogram becomes negative)
    decline_bars = n_bars // 2
    decline_prices = np.linspace(base_price, base_price - 5, decline_bars)

    # Recovery phase (histogram becomes positive and grows)
    recovery_bars = n_bars - decline_bars
    recovery_prices = np.linspace(base_price - 5, base_price + 5, recovery_bars)

    prices = np.concatenate([decline_prices, recovery_prices])
    prices += np.random.randn(n_bars) * 0.2

    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

    return pd.DataFrame({
        'open': prices - 0.2,
        'high': prices + 0.5,
        'low': prices - 0.5,
        'close': prices,
        'volume': [1_000_000] * n_bars
    }, index=dates)


def create_macd_bearish_crossover_df(
    n_bars: int = 60,
    base_price: float = 100.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create DataFrame that produces a MACD bearish crossover.

    Price pattern: rally followed by decline.
    """
    np.random.seed(seed)

    # Rally phase (histogram becomes positive)
    rally_bars = n_bars // 2
    rally_prices = np.linspace(base_price, base_price + 5, rally_bars)

    # Decline phase (histogram becomes negative and falls)
    decline_bars = n_bars - rally_bars
    decline_prices = np.linspace(base_price + 5, base_price - 5, decline_bars)

    prices = np.concatenate([rally_prices, decline_prices])
    prices += np.random.randn(n_bars) * 0.2

    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

    return pd.DataFrame({
        'open': prices + 0.2,
        'high': prices + 0.5,
        'low': prices - 0.5,
        'close': prices,
        'volume': [1_000_000] * n_bars
    }, index=dates)


def create_flat_price_df(
    n_bars: int = 20,
    price: float = 100.0,
    volume: int = 1_000_000
) -> pd.DataFrame:
    """
    Create DataFrame where high == low (edge case for stochastic).
    """
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

    return pd.DataFrame({
        'open': [price] * n_bars,
        'high': [price] * n_bars,
        'low': [price] * n_bars,
        'close': [price] * n_bars,
        'volume': [volume] * n_bars
    }, index=dates)


def create_orb_setup_df(
    orb_bars: int = 15,
    post_orb_bars: int = 35,
    orb_high: float = 100.50,
    orb_low: float = 99.50,
    base_volume: int = 1_000_000,
    seed: int = 42
) -> tuple[pd.DataFrame, dict]:
    """
    Create DataFrame simulating the ORB period followed by monitoring.

    Returns:
        Tuple of (DataFrame, dict with orb_high, orb_low, etc.)
    """
    np.random.seed(seed)
    n_bars = orb_bars + post_orb_bars

    # ORB period: consolidation within range
    orb_mid = (orb_high + orb_low) / 2
    orb_prices = orb_mid + np.random.randn(orb_bars) * 0.2
    orb_prices = np.clip(orb_prices, orb_low + 0.05, orb_high - 0.05)

    # Post-ORB: slight uptrend
    post_prices = np.linspace(orb_mid, orb_high + 0.5, post_orb_bars)
    post_prices += np.random.randn(post_orb_bars) * 0.1

    prices = np.concatenate([orb_prices, post_prices])

    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

    df = pd.DataFrame({
        'open': prices - 0.1,
        'high': np.concatenate([
            np.clip(prices[:orb_bars] + 0.2, None, orb_high),
            prices[orb_bars:] + 0.2
        ]),
        'low': np.concatenate([
            np.clip(prices[:orb_bars] - 0.2, orb_low, None),
            prices[orb_bars:] - 0.2
        ]),
        'close': prices,
        'volume': [base_volume] * orb_bars + [base_volume * 2] * post_orb_bars
    }, index=dates)

    orb_info = {
        'orb_high': orb_high,
        'orb_low': orb_low,
        'orb_range': orb_high - orb_low,
        'orb_bars': orb_bars,
        'post_orb_bars': post_orb_bars
    }

    return df, orb_info
