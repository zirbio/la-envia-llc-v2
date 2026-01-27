"""
Market scenario fixtures for testing signal generation.

Provides pre-configured scenarios that represent common trading situations:
- Strong breakout
- Weak breakout
- False breakout
- High volume spike
- MACD confirmation/rejection
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from strategy.orb import OpeningRange


@dataclass
class MarketScenario:
    """Complete market scenario for testing signal generation."""
    name: str
    description: str

    # Opening Range
    orb: OpeningRange

    # Current price data
    price: float
    vwap: float
    rsi: float
    rel_volume: float

    # MACD data
    macd_histogram: float
    prev_macd_histogram: Optional[float] = None

    # Sentiment
    sentiment: float = 0.0

    # Last candle close
    last_candle_close: Optional[float] = None

    # Expected outcome
    expected_direction: Optional[str] = None  # 'LONG', 'SHORT', or None
    expected_pass: bool = False  # Whether signal should be generated


def create_orb(
    symbol: str = "TEST",
    high: float = 100.0,
    low: float = 98.0,
    vwap: float = 99.0
) -> OpeningRange:
    """Helper to create OpeningRange."""
    return OpeningRange(
        symbol=symbol,
        high=high,
        low=low,
        range_size=high - low,
        vwap=vwap,
        timestamp=datetime.now()
    )


# ============================================================================
# LONG Breakout Scenarios
# ============================================================================

STRONG_LONG_BREAKOUT = MarketScenario(
    name="Strong LONG Breakout",
    description="All conditions perfect for LONG signal",
    orb=create_orb(high=100.0, low=98.0, vwap=99.0),
    price=100.50,
    vwap=99.00,
    rsi=50.0,
    rel_volume=3.0,
    macd_histogram=0.20,
    prev_macd_histogram=0.15,
    sentiment=0.5,
    last_candle_close=100.40,
    expected_direction='LONG',
    expected_pass=True
)

MODERATE_LONG_BREAKOUT = MarketScenario(
    name="Moderate LONG Breakout",
    description="Decent conditions for LONG signal",
    orb=create_orb(high=100.0, low=98.0, vwap=99.0),
    price=100.30,
    vwap=99.50,
    rsi=55.0,
    rel_volume=1.8,
    macd_histogram=0.10,
    prev_macd_histogram=0.08,
    sentiment=0.2,
    last_candle_close=100.25,
    expected_direction='LONG',
    expected_pass=True  # Should pass MODERATE level
)

WEAK_LONG_BREAKOUT = MarketScenario(
    name="Weak LONG Breakout",
    description="Marginal conditions - may not pass MODERATE",
    orb=create_orb(high=100.0, low=98.0, vwap=99.0),
    price=100.15,
    vwap=100.00,
    rsi=65.0,
    rel_volume=1.3,
    macd_histogram=0.05,
    prev_macd_histogram=0.04,
    sentiment=0.0,
    last_candle_close=100.12,
    expected_direction='LONG',
    expected_pass=False  # Score likely too low
)


# ============================================================================
# SHORT Breakout Scenarios
# ============================================================================

STRONG_SHORT_BREAKOUT = MarketScenario(
    name="Strong SHORT Breakout",
    description="All conditions perfect for SHORT signal",
    orb=create_orb(high=100.0, low=98.0, vwap=99.0),
    price=97.50,
    vwap=99.50,
    rsi=50.0,
    rel_volume=3.0,
    macd_histogram=-0.20,
    prev_macd_histogram=-0.15,
    sentiment=-0.5,
    last_candle_close=97.60,
    expected_direction='SHORT',
    expected_pass=True
)

MODERATE_SHORT_BREAKOUT = MarketScenario(
    name="Moderate SHORT Breakout",
    description="Decent conditions for SHORT signal",
    orb=create_orb(high=100.0, low=98.0, vwap=99.0),
    price=97.70,
    vwap=99.00,
    rsi=45.0,
    rel_volume=1.8,
    macd_histogram=-0.10,
    prev_macd_histogram=-0.08,
    sentiment=-0.2,
    last_candle_close=97.75,
    expected_direction='SHORT',
    expected_pass=True
)


# ============================================================================
# False Breakout Scenarios
# ============================================================================

FALSE_LONG_BREAKOUT_LOW_VOLUME = MarketScenario(
    name="False LONG Breakout - Low Volume",
    description="Price breaks but volume insufficient",
    orb=create_orb(high=100.0, low=98.0, vwap=99.0),
    price=100.50,
    vwap=99.50,
    rsi=50.0,
    rel_volume=1.1,  # Below 1.2 floor
    macd_histogram=0.15,
    prev_macd_histogram=0.12,
    sentiment=0.3,
    last_candle_close=100.40,
    expected_direction=None,
    expected_pass=False
)

FALSE_LONG_BREAKOUT_VWAP_FAIL = MarketScenario(
    name="False LONG Breakout - Below VWAP",
    description="Price breaks ORB but below VWAP",
    orb=create_orb(high=100.0, low=98.0, vwap=99.0),
    price=100.50,
    vwap=101.00,  # Price below VWAP
    rsi=50.0,
    rel_volume=2.0,
    macd_histogram=0.15,
    prev_macd_histogram=0.12,
    sentiment=0.3,
    last_candle_close=100.40,
    expected_direction=None,
    expected_pass=False
)

FALSE_LONG_BREAKOUT_RSI_OVERBOUGHT = MarketScenario(
    name="False LONG Breakout - RSI Overbought",
    description="Price breaks but RSI indicates overbought",
    orb=create_orb(high=100.0, low=98.0, vwap=99.0),
    price=100.50,
    vwap=99.50,
    rsi=78.0,  # Above overbought threshold
    rel_volume=2.0,
    macd_histogram=0.15,
    prev_macd_histogram=0.12,
    sentiment=0.3,
    last_candle_close=100.40,
    expected_direction=None,
    expected_pass=False
)

FALSE_LONG_BREAKOUT_NO_CANDLE_CONFIRM = MarketScenario(
    name="False LONG Breakout - Candle Not Confirmed",
    description="Price breaks but last candle closed below",
    orb=create_orb(high=100.0, low=98.0, vwap=99.0),
    price=100.50,
    vwap=99.50,
    rsi=50.0,
    rel_volume=2.0,
    macd_histogram=0.15,
    prev_macd_histogram=0.12,
    sentiment=0.3,
    last_candle_close=99.90,  # Below breakout level
    expected_direction=None,
    expected_pass=False
)


# ============================================================================
# Edge Case Scenarios
# ============================================================================

NO_BREAKOUT = MarketScenario(
    name="No Breakout",
    description="Price within ORB range",
    orb=create_orb(high=100.0, low=98.0, vwap=99.0),
    price=99.00,  # Within range
    vwap=99.00,
    rsi=50.0,
    rel_volume=2.0,
    macd_histogram=0.15,
    prev_macd_histogram=0.12,
    sentiment=0.3,
    last_candle_close=99.00,
    expected_direction=None,
    expected_pass=False
)

PRICE_IN_BUFFER_ZONE = MarketScenario(
    name="Price in Buffer Zone",
    description="Price above ORB high but below buffer",
    orb=create_orb(high=100.0, low=98.0, vwap=99.0),
    price=100.05,  # Above 100.00 but below 100.10 (0.1% buffer)
    vwap=99.50,
    rsi=50.0,
    rel_volume=2.0,
    macd_histogram=0.15,
    prev_macd_histogram=0.12,
    sentiment=0.3,
    last_candle_close=100.03,
    expected_direction=None,
    expected_pass=False
)

BEARISH_MACD_FOR_LONG = MarketScenario(
    name="Bearish MACD for LONG",
    description="Price breaks high but MACD is bearish",
    orb=create_orb(high=100.0, low=98.0, vwap=99.0),
    price=100.50,
    vwap=99.50,
    rsi=50.0,
    rel_volume=2.0,
    macd_histogram=-0.10,  # Negative histogram
    prev_macd_histogram=-0.05,
    sentiment=0.3,
    last_candle_close=100.40,
    expected_direction='LONG',  # May still pass with low MACD score
    expected_pass=True  # Depends on other factors
)

NEGATIVE_SENTIMENT_FOR_LONG = MarketScenario(
    name="Negative Sentiment for LONG",
    description="Price breaks high but sentiment very negative",
    orb=create_orb(high=100.0, low=98.0, vwap=99.0),
    price=100.50,
    vwap=99.50,
    rsi=50.0,
    rel_volume=2.0,
    macd_histogram=0.15,
    prev_macd_histogram=0.12,
    sentiment=-0.8,  # Very negative
    last_candle_close=100.40,
    expected_direction='LONG',  # May pass with low sentiment score
    expected_pass=True  # Depends on min_sentiment_long threshold
)


# ============================================================================
# Scenario Collections
# ============================================================================

ALL_SCENARIOS = [
    STRONG_LONG_BREAKOUT,
    MODERATE_LONG_BREAKOUT,
    WEAK_LONG_BREAKOUT,
    STRONG_SHORT_BREAKOUT,
    MODERATE_SHORT_BREAKOUT,
    FALSE_LONG_BREAKOUT_LOW_VOLUME,
    FALSE_LONG_BREAKOUT_VWAP_FAIL,
    FALSE_LONG_BREAKOUT_RSI_OVERBOUGHT,
    FALSE_LONG_BREAKOUT_NO_CANDLE_CONFIRM,
    NO_BREAKOUT,
    PRICE_IN_BUFFER_ZONE,
    BEARISH_MACD_FOR_LONG,
    NEGATIVE_SENTIMENT_FOR_LONG,
]

PASSING_SCENARIOS = [s for s in ALL_SCENARIOS if s.expected_pass]
FAILING_SCENARIOS = [s for s in ALL_SCENARIOS if not s.expected_pass]
LONG_SCENARIOS = [s for s in ALL_SCENARIOS if s.expected_direction == 'LONG']
SHORT_SCENARIOS = [s for s in ALL_SCENARIOS if s.expected_direction == 'SHORT']
