"""
Pytest configuration and shared fixtures for the Alpaca ORB Trading Bot tests.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass


# ============================================================================
# OHLCV DataFrame Fixtures
# ============================================================================

@pytest.fixture
def sample_ohlcv_df():
    """
    Generate a realistic 50-bar OHLCV DataFrame for testing indicators.
    Simulates price movement with volatility.
    """
    np.random.seed(42)  # Reproducibility
    n_bars = 50

    # Generate realistic price movement
    base_price = 100.0
    returns = np.random.randn(n_bars) * 0.02  # 2% daily volatility
    close_prices = base_price * np.cumprod(1 + returns)

    # Generate OHLC from close
    high_prices = close_prices * (1 + np.abs(np.random.randn(n_bars) * 0.01))
    low_prices = close_prices * (1 - np.abs(np.random.randn(n_bars) * 0.01))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    # Ensure high >= close >= low
    high_prices = np.maximum(high_prices, close_prices)
    low_prices = np.minimum(low_prices, close_prices)

    # Volume
    base_volume = 1_000_000
    volumes = (base_volume * (1 + np.random.randn(n_bars) * 0.3)).astype(int)
    volumes = np.maximum(volumes, 100_000)

    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)


@pytest.fixture
def flat_price_df():
    """
    DataFrame where high == low (edge case for stochastic division by zero).
    """
    n_bars = 20
    price = 100.0

    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

    return pd.DataFrame({
        'open': [price] * n_bars,
        'high': [price] * n_bars,
        'low': [price] * n_bars,
        'close': [price] * n_bars,
        'volume': [1_000_000] * n_bars
    }, index=dates)


@pytest.fixture
def trending_up_df():
    """DataFrame with a clear uptrend for RSI testing."""
    n_bars = 30
    prices = np.linspace(100, 150, n_bars)  # Clear uptrend

    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

    return pd.DataFrame({
        'open': prices - 0.5,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices,
        'volume': [1_000_000] * n_bars
    }, index=dates)


@pytest.fixture
def trending_down_df():
    """DataFrame with a clear downtrend for RSI testing."""
    n_bars = 30
    prices = np.linspace(150, 100, n_bars)  # Clear downtrend

    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

    return pd.DataFrame({
        'open': prices + 0.5,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices,
        'volume': [1_000_000] * n_bars
    }, index=dates)


@pytest.fixture
def volume_spike_df():
    """DataFrame with a volume spike on the last bar."""
    n_bars = 20
    np.random.seed(42)

    prices = np.linspace(100, 110, n_bars)
    volumes = [500_000] * (n_bars - 1) + [2_500_000]  # 5x spike on last bar

    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

    return pd.DataFrame({
        'open': prices - 0.5,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices,
        'volume': volumes
    }, index=dates)


# ============================================================================
# Opening Range Fixtures
# ============================================================================

@pytest.fixture
def sample_opening_range():
    """Sample OpeningRange for ORB strategy tests."""
    from strategy.orb import OpeningRange

    return OpeningRange(
        symbol="AAPL",
        high=150.50,
        low=148.50,
        range_size=2.0,
        vwap=149.50,
        timestamp=datetime.now()
    )


@pytest.fixture
def wide_opening_range():
    """Opening range with wider spread."""
    from strategy.orb import OpeningRange

    return OpeningRange(
        symbol="TSLA",
        high=255.00,
        low=245.00,
        range_size=10.0,
        vwap=250.00,
        timestamp=datetime.now()
    )


# ============================================================================
# Trade Signal Fixtures
# ============================================================================

@pytest.fixture
def long_trade_signal():
    """Sample LONG trade signal."""
    from strategy.orb import TradeSignal, SignalType

    return TradeSignal(
        symbol="AAPL",
        signal_type=SignalType.LONG,
        entry_price=151.00,
        stop_loss=148.50,
        take_profit=156.00,
        position_size=100,
        risk_amount=250.0,
        orb_high=150.50,
        orb_low=148.50,
        vwap=149.50,
        rsi=55.0,
        relative_volume=1.8,
        timestamp=datetime.now(),
        macd=0.5,
        macd_signal=0.3,
        macd_histogram=0.2,
        sentiment_score=0.3
    )


@pytest.fixture
def short_trade_signal():
    """Sample SHORT trade signal."""
    from strategy.orb import TradeSignal, SignalType

    return TradeSignal(
        symbol="TSLA",
        signal_type=SignalType.SHORT,
        entry_price=244.00,
        stop_loss=255.00,
        take_profit=222.00,
        position_size=50,
        risk_amount=550.0,
        orb_high=255.00,
        orb_low=245.00,
        vwap=250.00,
        rsi=45.0,
        relative_volume=2.0,
        timestamp=datetime.now(),
        macd=-0.5,
        macd_signal=-0.3,
        macd_histogram=-0.2,
        sentiment_score=-0.4
    )


# ============================================================================
# Scan Result Fixtures
# ============================================================================

@pytest.fixture
def sample_scan_result():
    """Sample ScanResult for premarket scanner tests."""
    from scanner.premarket import ScanResult

    return ScanResult(
        symbol="NVDA",
        prev_close=450.0,
        current_price=470.0,
        gap_percent=4.44,
        premarket_volume=800_000,
        avg_daily_volume=20_000_000,
        score=75.0,
        sentiment_score=0.5,
        sentiment_label="Alcista",
        sentiment_news_count=10
    )


# ============================================================================
# Sentiment Fixtures
# ============================================================================

@pytest.fixture
def sample_sentiment_result():
    """Sample SentimentResult."""
    from data.sentiment import SentimentResult

    return SentimentResult(
        symbol="AAPL",
        score=0.4,
        news_count=15,
        positive_count=8,
        negative_count=4,
        neutral_count=3,
        headlines=["Apple beats expectations", "Strong iPhone sales"],
        timestamp=datetime.now()
    )


@pytest.fixture
def bearish_news():
    """List of bearish news articles for sentiment analysis."""
    return [
        {"headline": "Stock crashes after earnings miss", "summary": "Company reports loss"},
        {"headline": "Investors sell off shares amid concerns", "summary": "Risk of decline"},
        {"headline": "Weak guidance leads to downgrade", "summary": "Analysts cut price targets"},
    ]


@pytest.fixture
def bullish_news():
    """List of bullish news articles for sentiment analysis."""
    return [
        {"headline": "Stock surges on strong earnings beat", "summary": "Record profits"},
        {"headline": "Company announces breakthrough product", "summary": "Bullish momentum"},
        {"headline": "Analysts upgrade stock citing growth", "summary": "Buy rating issued"},
    ]


@pytest.fixture
def neutral_news():
    """List of neutral news articles for sentiment analysis."""
    return [
        {"headline": "Company reports quarterly results", "summary": "Mixed performance"},
        {"headline": "CEO discusses market conditions", "summary": "Industry update"},
        {"headline": "Regulatory filing submitted", "summary": "Standard compliance"},
    ]


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_market_data(mocker):
    """Mock MarketDataClient for isolated testing."""
    mock = mocker.patch('data.market_data.market_data')

    # Configure default return values
    mock.get_bars.return_value = pd.DataFrame()
    mock.get_premarket_data.return_value = None
    mock.get_avg_daily_volume.return_value = 1_000_000

    return mock


@pytest.fixture
def mock_alpaca_client(mocker):
    """Mock Alpaca TradingClient."""
    mock_client = MagicMock()

    # Mock account
    mock_account = MagicMock()
    mock_account.equity = "25000.0"
    mock_account.cash = "10000.0"
    mock_account.buying_power = "50000.0"
    mock_account.daytrade_count = 0
    mock_account.pattern_day_trader = False
    mock_client.get_account.return_value = mock_account

    # Mock clock
    mock_clock = MagicMock()
    mock_clock.is_open = True
    mock_clock.next_open = datetime.now() + timedelta(hours=12)
    mock_clock.next_close = datetime.now() + timedelta(hours=6)
    mock_client.get_clock.return_value = mock_clock

    # Mock positions
    mock_client.get_all_positions.return_value = []

    return mock_client


@pytest.fixture
def mock_aiohttp_session(mocker):
    """Mock aiohttp ClientSession for async API calls."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=[])
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    mocker.patch('aiohttp.ClientSession', return_value=mock_session)

    return mock_session


# ============================================================================
# Trade History Fixtures
# ============================================================================

@pytest.fixture
def sample_trade_history():
    """Sample trade history for Kelly calculation tests."""
    from strategy.orb import TradeResult

    return [
        TradeResult(symbol="AAPL", entry_price=150, exit_price=155, pnl=5, pnl_pct=0.033, won=True),
        TradeResult(symbol="TSLA", entry_price=250, exit_price=245, pnl=-5, pnl_pct=-0.02, won=False),
        TradeResult(symbol="NVDA", entry_price=400, exit_price=420, pnl=20, pnl_pct=0.05, won=True),
        TradeResult(symbol="AMD", entry_price=100, exit_price=105, pnl=5, pnl_pct=0.05, won=True),
        TradeResult(symbol="GOOGL", entry_price=140, exit_price=138, pnl=-2, pnl_pct=-0.014, won=False),
        TradeResult(symbol="META", entry_price=300, exit_price=315, pnl=15, pnl_pct=0.05, won=True),
    ]


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def mock_settings(mocker):
    """Mock settings for isolated testing."""
    from config.settings import TradingConfig, SentimentConfig

    mock_trading = TradingConfig()
    mock_sentiment = SentimentConfig()

    mock = mocker.patch('config.settings.settings')
    mock.trading = mock_trading
    mock.sentiment = mock_sentiment

    return mock


# ============================================================================
# Helper Functions
# ============================================================================

def create_price_series(values: list[float]) -> pd.Series:
    """Create a price series from a list of values."""
    return pd.Series(values, index=range(len(values)))


def create_ohlcv_bar(open_price: float, high: float, low: float, close: float, volume: int) -> dict:
    """Create a single OHLCV bar."""
    return {
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }
