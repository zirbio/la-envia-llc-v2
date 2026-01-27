"""
Unit tests for _check_long_conditions() and _check_short_conditions() in strategy/orb.py.

These functions check all 6 conditions for generating LONG/SHORT signals:
1. Price breakout (above/below ORB)
2. VWAP alignment (price vs VWAP)
3. Volume (relative volume threshold)
4. RSI (not overbought/oversold)
5. MACD (bullish/bearish)
6. Sentiment (above/below threshold)

Run with: pytest tests/unit/test_signal_conditions.py -v
"""
import pytest
from datetime import datetime
from unittest.mock import patch

from strategy.orb import ORBStrategy, OpeningRange
from config.settings import TradingConfig, SentimentConfig, SignalLevel


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def orb_strategy():
    """Create fresh ORBStrategy with MODERATE level."""
    with patch('strategy.orb.settings') as mock_settings:
        config = TradingConfig()
        config.signal_level = SignalLevel.MODERATE
        mock_settings.trading = config
        mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
        strategy = ORBStrategy()
    return strategy


@pytest.fixture
def sample_orb():
    """Create a sample OpeningRange for testing."""
    return OpeningRange(
        symbol="TEST",
        high=100.00,
        low=98.00,
        range_size=2.00,
        vwap=99.00,
        timestamp=datetime.now()
    )


# ============================================================================
# LONG Condition 1: Price Breakout
# ============================================================================

@pytest.mark.unit
class TestLongPriceBreakout:
    """Tests for LONG price breakout condition."""

    def test_price_above_orb_high_plus_buffer(self, orb_strategy, sample_orb):
        """Price above ORB high + buffer should pass."""
        buffer = orb_strategy.config.breakout_buffer_pct  # 0.001
        breakout_level = sample_orb.high * (1 + buffer)  # 100.10

        result = orb_strategy._check_long_conditions(
            price=100.20,  # Above 100.10
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.5,
            macd_bullish=True,
            sentiment=0.0
        )

        assert result is True

    def test_price_below_orb_high_fails(self, orb_strategy, sample_orb):
        """Price below ORB high should fail."""
        result = orb_strategy._check_long_conditions(
            price=99.50,  # Below ORB high (100.00)
            orb=sample_orb,
            vwap=98.00,  # Satisfy other conditions
            rsi=50.0,
            rel_volume=1.5,
            macd_bullish=True,
            sentiment=0.0
        )

        assert result is False

    def test_price_in_buffer_zone_fails(self, orb_strategy, sample_orb):
        """Price in buffer zone (above ORB but below buffer) should fail."""
        buffer = orb_strategy.config.breakout_buffer_pct
        breakout_level = sample_orb.high * (1 + buffer)  # 100.10

        result = orb_strategy._check_long_conditions(
            price=100.05,  # Above 100.00 but below 100.10
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.5,
            macd_bullish=True,
            sentiment=0.0
        )

        assert result is False


# ============================================================================
# LONG Condition 2: VWAP Alignment
# ============================================================================

@pytest.mark.unit
class TestLongVWAPAlignment:
    """Tests for LONG VWAP alignment condition."""

    def test_price_above_vwap_passes(self, orb_strategy, sample_orb):
        """Price above VWAP should pass."""
        result = orb_strategy._check_long_conditions(
            price=100.20,
            orb=sample_orb,
            vwap=99.00,  # Price > VWAP
            rsi=50.0,
            rel_volume=1.5,
            macd_bullish=True,
            sentiment=0.0
        )

        assert result is True

    def test_price_below_vwap_fails(self, orb_strategy, sample_orb):
        """Price below VWAP should fail for LONG."""
        result = orb_strategy._check_long_conditions(
            price=100.20,  # Above ORB
            orb=sample_orb,
            vwap=101.00,  # Price < VWAP
            rsi=50.0,
            rel_volume=1.5,
            macd_bullish=True,
            sentiment=0.0
        )

        assert result is False

    def test_price_equal_vwap_fails(self, orb_strategy, sample_orb):
        """Price equal to VWAP should fail (need price > VWAP)."""
        result = orb_strategy._check_long_conditions(
            price=100.20,
            orb=sample_orb,
            vwap=100.20,  # Price == VWAP
            rsi=50.0,
            rel_volume=1.5,
            macd_bullish=True,
            sentiment=0.0
        )

        assert result is False


# ============================================================================
# LONG Condition 3: Relative Volume
# ============================================================================

@pytest.mark.unit
class TestLongRelativeVolume:
    """Tests for LONG relative volume condition."""

    def test_volume_above_min_passes(self, orb_strategy, sample_orb):
        """Volume above min_relative_volume should pass."""
        min_vol = orb_strategy.config.min_relative_volume  # 1.2 for MODERATE

        result = orb_strategy._check_long_conditions(
            price=100.20,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=min_vol + 0.1,  # Above minimum
            macd_bullish=True,
            sentiment=0.0
        )

        assert result is True

    def test_volume_below_min_fails(self, orb_strategy, sample_orb):
        """Volume below min_relative_volume should fail."""
        min_vol = orb_strategy.config.min_relative_volume  # 1.2

        result = orb_strategy._check_long_conditions(
            price=100.20,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=min_vol - 0.01,  # Below minimum
            macd_bullish=True,
            sentiment=0.0
        )

        assert result is False

    def test_volume_at_min_passes(self, orb_strategy, sample_orb):
        """Volume exactly at min_relative_volume should pass."""
        min_vol = orb_strategy.config.min_relative_volume  # 1.2

        result = orb_strategy._check_long_conditions(
            price=100.20,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=min_vol,  # Exactly at minimum
            macd_bullish=True,
            sentiment=0.0
        )

        assert result is True


# ============================================================================
# LONG Condition 4: RSI Not Overbought
# ============================================================================

@pytest.mark.unit
class TestLongRSI:
    """Tests for LONG RSI condition."""

    def test_rsi_below_overbought_passes(self, orb_strategy, sample_orb):
        """RSI below overbought threshold should pass."""
        overbought = orb_strategy.config.rsi_overbought  # 75 for MODERATE

        result = orb_strategy._check_long_conditions(
            price=100.20,
            orb=sample_orb,
            vwap=99.00,
            rsi=overbought - 1,  # 74
            rel_volume=1.5,
            macd_bullish=True,
            sentiment=0.0
        )

        assert result is True

    def test_rsi_at_overbought_fails(self, orb_strategy, sample_orb):
        """RSI at overbought threshold should fail."""
        overbought = orb_strategy.config.rsi_overbought  # 75

        result = orb_strategy._check_long_conditions(
            price=100.20,
            orb=sample_orb,
            vwap=99.00,
            rsi=overbought,  # 75 (not < 75)
            rel_volume=1.5,
            macd_bullish=True,
            sentiment=0.0
        )

        assert result is False

    def test_rsi_above_overbought_fails(self, orb_strategy, sample_orb):
        """RSI above overbought threshold should fail."""
        overbought = orb_strategy.config.rsi_overbought  # 75

        result = orb_strategy._check_long_conditions(
            price=100.20,
            orb=sample_orb,
            vwap=99.00,
            rsi=overbought + 5,  # 80
            rel_volume=1.5,
            macd_bullish=True,
            sentiment=0.0
        )

        assert result is False


# ============================================================================
# LONG Condition 5: MACD Bullish
# ============================================================================

@pytest.mark.unit
class TestLongMACD:
    """Tests for LONG MACD condition."""

    def test_macd_bullish_passes(self, orb_strategy, sample_orb):
        """MACD bullish should pass."""
        result = orb_strategy._check_long_conditions(
            price=100.20,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.5,
            macd_bullish=True,
            sentiment=0.0
        )

        assert result is True

    def test_macd_not_bullish_fails(self, orb_strategy, sample_orb):
        """MACD not bullish should fail."""
        result = orb_strategy._check_long_conditions(
            price=100.20,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.5,
            macd_bullish=False,  # Not bullish
            sentiment=0.0
        )

        assert result is False


# ============================================================================
# LONG Condition 6: Sentiment
# ============================================================================

@pytest.mark.unit
class TestLongSentiment:
    """Tests for LONG sentiment condition."""

    def test_sentiment_above_min_passes(self, orb_strategy, sample_orb):
        """Sentiment above min_sentiment_long should pass."""
        min_sent = orb_strategy.config.signal_config.min_sentiment_long  # -0.5

        result = orb_strategy._check_long_conditions(
            price=100.20,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.5,
            macd_bullish=True,
            sentiment=min_sent + 0.1  # Above minimum
        )

        assert result is True

    def test_sentiment_below_min_fails(self, orb_strategy, sample_orb):
        """Sentiment below min_sentiment_long should fail."""
        min_sent = orb_strategy.config.signal_config.min_sentiment_long  # -0.5

        result = orb_strategy._check_long_conditions(
            price=100.20,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.5,
            macd_bullish=True,
            sentiment=min_sent - 0.1  # Below minimum
        )

        assert result is False

    def test_sentiment_at_min_passes(self, orb_strategy, sample_orb):
        """Sentiment exactly at min_sentiment_long should pass."""
        min_sent = orb_strategy.config.signal_config.min_sentiment_long  # -0.5

        result = orb_strategy._check_long_conditions(
            price=100.20,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.5,
            macd_bullish=True,
            sentiment=min_sent  # Exactly at minimum
        )

        assert result is True


# ============================================================================
# SHORT Condition Tests
# ============================================================================

@pytest.mark.unit
class TestShortPriceBreakout:
    """Tests for SHORT price breakout condition."""

    def test_price_below_orb_low_minus_buffer(self, orb_strategy, sample_orb):
        """Price below ORB low - buffer should pass."""
        buffer = orb_strategy.config.breakout_buffer_pct  # 0.001
        breakout_level = sample_orb.low * (1 - buffer)  # ~97.902

        result = orb_strategy._check_short_conditions(
            price=97.80,  # Below 97.902
            orb=sample_orb,
            vwap=99.00,  # Price < VWAP
            rsi=50.0,
            rel_volume=1.5,
            macd_bearish=True,
            sentiment=0.0
        )

        assert result is True

    def test_price_above_orb_low_fails(self, orb_strategy, sample_orb):
        """Price above ORB low should fail."""
        result = orb_strategy._check_short_conditions(
            price=98.50,  # Above ORB low (98.00)
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.5,
            macd_bearish=True,
            sentiment=0.0
        )

        assert result is False


@pytest.mark.unit
class TestShortVWAPAlignment:
    """Tests for SHORT VWAP alignment condition."""

    def test_price_below_vwap_passes(self, orb_strategy, sample_orb):
        """Price below VWAP should pass for SHORT."""
        result = orb_strategy._check_short_conditions(
            price=97.80,
            orb=sample_orb,
            vwap=99.00,  # Price < VWAP
            rsi=50.0,
            rel_volume=1.5,
            macd_bearish=True,
            sentiment=0.0
        )

        assert result is True

    def test_price_above_vwap_fails(self, orb_strategy, sample_orb):
        """Price above VWAP should fail for SHORT."""
        result = orb_strategy._check_short_conditions(
            price=97.80,
            orb=sample_orb,
            vwap=97.00,  # Price > VWAP
            rsi=50.0,
            rel_volume=1.5,
            macd_bearish=True,
            sentiment=0.0
        )

        assert result is False


@pytest.mark.unit
class TestShortRSI:
    """Tests for SHORT RSI condition."""

    def test_rsi_above_oversold_passes(self, orb_strategy, sample_orb):
        """RSI above oversold threshold should pass."""
        oversold = orb_strategy.config.rsi_oversold  # 25 for MODERATE

        result = orb_strategy._check_short_conditions(
            price=97.80,
            orb=sample_orb,
            vwap=99.00,
            rsi=oversold + 1,  # 26
            rel_volume=1.5,
            macd_bearish=True,
            sentiment=0.0
        )

        assert result is True

    def test_rsi_at_oversold_fails(self, orb_strategy, sample_orb):
        """RSI at oversold threshold should fail."""
        oversold = orb_strategy.config.rsi_oversold  # 25

        result = orb_strategy._check_short_conditions(
            price=97.80,
            orb=sample_orb,
            vwap=99.00,
            rsi=oversold,  # 25 (not > 25)
            rel_volume=1.5,
            macd_bearish=True,
            sentiment=0.0
        )

        assert result is False


@pytest.mark.unit
class TestShortSentiment:
    """Tests for SHORT sentiment condition."""

    def test_sentiment_below_max_passes(self, orb_strategy, sample_orb):
        """Sentiment below max_sentiment_short should pass."""
        max_sent = orb_strategy.config.signal_config.max_sentiment_short  # 0.5

        result = orb_strategy._check_short_conditions(
            price=97.80,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.5,
            macd_bearish=True,
            sentiment=max_sent - 0.1  # Below maximum
        )

        assert result is True

    def test_sentiment_above_max_fails(self, orb_strategy, sample_orb):
        """Sentiment above max_sentiment_short should fail."""
        max_sent = orb_strategy.config.signal_config.max_sentiment_short  # 0.5

        result = orb_strategy._check_short_conditions(
            price=97.80,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.5,
            macd_bearish=True,
            sentiment=max_sent + 0.1  # Above maximum
        )

        assert result is False


# ============================================================================
# Candle Close Confirmation Tests
# ============================================================================

@pytest.mark.unit
class TestCandleCloseConfirmation:
    """Tests for candle close confirmation in conditions."""

    def test_long_candle_close_required(self, orb_strategy, sample_orb):
        """LONG should fail if candle close doesn't confirm."""
        result = orb_strategy._check_long_conditions(
            price=100.20,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.5,
            macd_bullish=True,
            sentiment=0.0,
            last_candle_close=99.90  # Below breakout level
        )

        assert result is False

    def test_long_candle_close_confirms(self, orb_strategy, sample_orb):
        """LONG should pass if candle close confirms."""
        result = orb_strategy._check_long_conditions(
            price=100.20,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.5,
            macd_bullish=True,
            sentiment=0.0,
            last_candle_close=100.15  # Above breakout level
        )

        assert result is True

    def test_short_candle_close_required(self, orb_strategy, sample_orb):
        """SHORT should fail if candle close doesn't confirm."""
        result = orb_strategy._check_short_conditions(
            price=97.80,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.5,
            macd_bearish=True,
            sentiment=0.0,
            last_candle_close=98.10  # Above breakout level
        )

        assert result is False


# ============================================================================
# All Conditions Combined
# ============================================================================

@pytest.mark.unit
class TestAllConditionsCombined:
    """Tests for all conditions working together."""

    def test_all_long_conditions_met(self, orb_strategy, sample_orb):
        """Should return True when ALL conditions are met."""
        result = orb_strategy._check_long_conditions(
            price=100.20,       # Above ORB + buffer
            orb=sample_orb,
            vwap=99.00,         # Price > VWAP
            rsi=50.0,           # Below overbought (75)
            rel_volume=1.5,     # Above min (1.2)
            macd_bullish=True,  # Bullish
            sentiment=0.0,      # Above min (-0.5)
            last_candle_close=100.15
        )

        assert result is True

    def test_one_long_condition_fails(self, orb_strategy, sample_orb):
        """Should return False when ANY single condition fails."""
        # Fail each condition one at a time
        base_kwargs = dict(
            price=100.20,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.5,
            macd_bullish=True,
            sentiment=0.0,
            last_candle_close=100.15
        )

        # Fail price
        result = orb_strategy._check_long_conditions(**{**base_kwargs, 'price': 99.50})
        assert result is False

        # Fail VWAP
        result = orb_strategy._check_long_conditions(**{**base_kwargs, 'vwap': 101.00})
        assert result is False

        # Fail RSI
        result = orb_strategy._check_long_conditions(**{**base_kwargs, 'rsi': 80.0})
        assert result is False

        # Fail volume
        result = orb_strategy._check_long_conditions(**{**base_kwargs, 'rel_volume': 1.0})
        assert result is False

        # Fail MACD
        result = orb_strategy._check_long_conditions(**{**base_kwargs, 'macd_bullish': False})
        assert result is False

        # Fail sentiment
        result = orb_strategy._check_long_conditions(**{**base_kwargs, 'sentiment': -0.6})
        assert result is False
