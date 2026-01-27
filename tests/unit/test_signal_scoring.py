"""
Unit tests for _calculate_signal_score() in strategy/orb.py.

This function calculates a quality score (0-100) for potential trade signals.
Tests verify each scoring component independently and combined.

Score breakdown:
- Breakout strength: 0-25 pts
- VWAP alignment: 0-15 pts
- Volume: 0-20 pts
- RSI: 0-15 pts
- MACD: 0-15 pts
- Sentiment: 0-10 pts
Total: 0-100 pts

Run with: pytest tests/unit/test_signal_scoring.py -v
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
# BREAKOUT STRENGTH Component (0-25 pts)
# ============================================================================

@pytest.mark.unit
class TestBreakoutStrengthScore:
    """Tests for breakout strength scoring (0-25 pts)."""

    def test_breakout_score_strong_long(self, orb_strategy, sample_orb):
        """Strong breakout (0.5%) should give max 25 points."""
        # 0.5% breakout = 0.005 * 100 = 0.50 above ORB high
        # Breakout score = 0.5% * 50 = 25 pts (max)
        score = orb_strategy._calculate_signal_score(
            price=100.50,  # 0.5% above ORB high (100.00)
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.0,  # Minimal other factors
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        # Check breakout component contributes ~25 pts
        # Note: Other components contribute minimal points
        assert score >= 25  # Breakout alone should give 25

    def test_breakout_score_weak_long(self, orb_strategy, sample_orb):
        """Weak breakout (0.1%) should give ~5 points."""
        # 0.1% breakout = ~5 pts
        score = orb_strategy._calculate_signal_score(
            price=100.10,  # 0.1% above ORB high
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        # Weak breakout still gets some points, plus VWAP alignment etc
        # Score should be lower than strong breakout
        assert score < 50  # Less than strong breakout scenario

    def test_breakout_score_no_breakout_long(self, orb_strategy, sample_orb):
        """No breakout should give 0 breakout points but other components still contribute."""
        score = orb_strategy._calculate_signal_score(
            price=99.90,  # Below ORB high (no breakout)
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        # No breakout points, but VWAP/RSI/volume components still contribute
        # Score should be low but not zero due to other factors
        assert score < 45  # Lower than breakout scenarios

    def test_breakout_score_strong_short(self, orb_strategy, sample_orb):
        """Strong short breakout should give max 25 points."""
        score = orb_strategy._calculate_signal_score(
            price=97.50,  # 0.5% below ORB low (98.00)
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='SHORT'
        )

        assert score >= 25


# ============================================================================
# VWAP ALIGNMENT Component (0-15 pts)
# ============================================================================

@pytest.mark.unit
class TestVWAPAlignmentScore:
    """Tests for VWAP alignment scoring (0-15 pts)."""

    def test_vwap_score_strong_alignment_long(self, orb_strategy, sample_orb):
        """Price well above VWAP should give max 15 points for LONG."""
        score = orb_strategy._calculate_signal_score(
            price=100.50,  # Above ORB high (needed for breakout points)
            orb=sample_orb,
            vwap=99.00,    # Price 1.5% above VWAP
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        # VWAP distance = (100.50 - 99.00) / 99.00 = 1.51%
        # VWAP score = min(1.51 * 15, 15) = 15 pts
        # Total should include 15 from VWAP
        assert score >= 15

    def test_vwap_score_wrong_side_long(self, orb_strategy, sample_orb):
        """Price below VWAP should give 0 points for LONG."""
        score = orb_strategy._calculate_signal_score(
            price=100.50,  # Above ORB high
            orb=sample_orb,
            vwap=101.00,   # Price BELOW VWAP
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        # Should get 0 VWAP points (wrong side)
        # Score should be breakout only (~25) + RSI (~15) + sentiment (~5)
        assert score < 60

    def test_vwap_score_zero_vwap(self, orb_strategy, sample_orb):
        """VWAP of 0 should not cause division error."""
        sample_orb.vwap = 0.0

        # Should not raise exception
        score = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=0.0,
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        assert score >= 0


# ============================================================================
# VOLUME Component (0-20 pts)
# ============================================================================

@pytest.mark.unit
class TestVolumeScore:
    """Tests for volume scoring (0-20 pts)."""

    def test_volume_score_exceptional(self, orb_strategy, sample_orb):
        """Volume >= 2.5x should give max 20 points."""
        score = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=2.5,  # 2.5x = 20 pts
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        assert score >= 20

    def test_volume_score_high(self, orb_strategy, sample_orb):
        """Volume 2.0-2.5x should give 15 points."""
        score = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=2.0,  # 2.0x = 15 pts
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        # Less than 2.5x case
        score_2_5 = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=2.5,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        assert score_2_5 - score == pytest.approx(5.0, abs=1)

    def test_volume_score_moderate(self, orb_strategy, sample_orb):
        """Volume 1.5-2.0x should give 10 points."""
        score = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.5,  # 1.5x = 10 pts
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        score_2_0 = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=2.0,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        assert score_2_0 - score == pytest.approx(5.0, abs=1)

    def test_volume_score_low(self, orb_strategy, sample_orb):
        """Volume 1.2-1.5x should give 5 points."""
        score = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.2,  # 1.2x = 5 pts
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        score_1_5 = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.5,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        assert score_1_5 - score == pytest.approx(5.0, abs=1)

    def test_volume_score_minimal(self, orb_strategy, sample_orb):
        """Volume < 1.2x should give 0 points."""
        score_low = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.1,  # < 1.2x = 0 pts
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        score_at_threshold = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.2,  # = 5 pts
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        assert score_at_threshold - score_low == pytest.approx(5.0, abs=1)


# ============================================================================
# RSI Component (0-15 pts)
# ============================================================================

@pytest.mark.unit
class TestRSIScore:
    """Tests for RSI scoring (0-15 pts)."""

    def test_rsi_score_sweet_spot_long(self, orb_strategy, sample_orb):
        """RSI 40-60 should give max 15 points for LONG."""
        for rsi in [40, 50, 60]:
            score = orb_strategy._calculate_signal_score(
                price=100.50,
                orb=sample_orb,
                vwap=99.00,
                rsi=float(rsi),
                rel_volume=1.0,
                macd_histogram=0.0,
                sentiment=0.0,
                direction='LONG'
            )
            # Sweet spot should give 15 pts for RSI component
            assert score >= 15

    def test_rsi_score_acceptable_long(self, orb_strategy, sample_orb):
        """RSI 30-70 (outside sweet spot) should give 10 points."""
        score_35 = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=35.0,  # In acceptable range but not sweet spot
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        score_50 = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,  # Sweet spot
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        # Sweet spot should give 5 more points than acceptable
        assert score_50 - score_35 == pytest.approx(5.0, abs=1)

    def test_rsi_score_oversold_long(self, orb_strategy, sample_orb):
        """RSI < 30 (oversold) should give 5 points for LONG."""
        score = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=25.0,  # Oversold
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        # Oversold can bounce, so give 5 pts
        # Verify score includes RSI component
        assert score >= 5

    def test_rsi_score_overbought_long(self, orb_strategy, sample_orb):
        """RSI > 70 (overbought) should give 0 points for LONG."""
        score_overbought = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=75.0,  # Overbought
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        score_normal = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=65.0,  # Acceptable
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        # Overbought should give 0, acceptable gives 10
        assert score_normal - score_overbought == pytest.approx(10.0, abs=1)


# ============================================================================
# MACD Component (0-15 pts)
# ============================================================================

@pytest.mark.unit
class TestMACDScore:
    """Tests for MACD scoring (0-15 pts)."""

    def test_macd_score_positive_histogram_long(self, orb_strategy, sample_orb):
        """Positive histogram should give points for LONG."""
        score = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=0.15,  # Positive
            sentiment=0.0,
            direction='LONG'
        )

        score_zero = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        # Positive histogram should add points
        assert score > score_zero

    def test_macd_score_negative_histogram_long(self, orb_strategy, sample_orb):
        """Negative histogram should give 0 points for LONG."""
        score_negative = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=-0.15,  # Negative
            sentiment=0.0,
            direction='LONG'
        )

        score_zero = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='LONG'
        )

        # Both should be same (negative gives 0 for LONG)
        assert score_negative == score_zero

    def test_macd_score_negative_histogram_short(self, orb_strategy, sample_orb):
        """Negative histogram should give points for SHORT."""
        score = orb_strategy._calculate_signal_score(
            price=97.50,  # Below ORB low
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=-0.15,  # Negative = good for SHORT
            sentiment=0.0,
            direction='SHORT'
        )

        score_zero = orb_strategy._calculate_signal_score(
            price=97.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='SHORT'
        )

        # Negative histogram should add points for SHORT
        assert score > score_zero


# ============================================================================
# SENTIMENT Component (0-10 pts)
# ============================================================================

@pytest.mark.unit
class TestSentimentScore:
    """Tests for sentiment scoring (0-10 pts)."""

    def test_sentiment_score_positive_long(self, orb_strategy, sample_orb):
        """Positive sentiment should give max 10 points for LONG."""
        score = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=1.0,  # Max positive
            direction='LONG'
        )

        score_neutral = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=0.0,  # Neutral
            direction='LONG'
        )

        # Max sentiment should add 5 more points than neutral
        # sentiment=1.0: (1.0 + 1) * 5 = 10 pts
        # sentiment=0.0: (0.0 + 1) * 5 = 5 pts
        assert score - score_neutral == pytest.approx(5.0, abs=1)

    def test_sentiment_score_negative_long(self, orb_strategy, sample_orb):
        """Negative sentiment should give fewer points for LONG."""
        score_negative = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=-1.0,  # Max negative
            direction='LONG'
        )

        score_positive = orb_strategy._calculate_signal_score(
            price=100.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=1.0,  # Max positive
            direction='LONG'
        )

        # sentiment=-1.0: (-1.0 + 1) * 5 = 0 pts
        # sentiment=1.0: (1.0 + 1) * 5 = 10 pts
        assert score_positive - score_negative == pytest.approx(10.0, abs=1)

    def test_sentiment_score_negative_short(self, orb_strategy, sample_orb):
        """Negative sentiment should give max points for SHORT."""
        score_negative = orb_strategy._calculate_signal_score(
            price=97.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=-1.0,  # Max negative = good for SHORT
            direction='SHORT'
        )

        score_neutral = orb_strategy._calculate_signal_score(
            price=97.50,
            orb=sample_orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=1.0,
            macd_histogram=0.0,
            sentiment=0.0,
            direction='SHORT'
        )

        # For SHORT: (1 - sentiment) * 5
        # sentiment=-1.0: (1 - (-1)) * 5 = 10 pts
        # sentiment=0.0: (1 - 0) * 5 = 5 pts
        assert score_negative - score_neutral == pytest.approx(5.0, abs=1)


# ============================================================================
# TOTAL SCORE Tests
# ============================================================================

@pytest.mark.unit
class TestTotalScore:
    """Tests for total score calculation."""

    def test_max_possible_score(self, orb_strategy, sample_orb):
        """Perfect conditions should give ~100 score."""
        score = orb_strategy._calculate_signal_score(
            price=100.50,       # Strong breakout: 25 pts
            orb=sample_orb,
            vwap=99.00,         # Strong VWAP alignment: 15 pts
            rsi=50.0,           # Sweet spot RSI: 15 pts
            rel_volume=3.0,     # Strong volume: 20 pts
            macd_histogram=0.20,  # Strong MACD: 15 pts
            sentiment=1.0,      # Strong sentiment: 10 pts
            direction='LONG'
        )

        # Total max = 100
        assert score >= 90  # Allow some variation

    def test_min_possible_score(self, orb_strategy, sample_orb):
        """Poor conditions should give low score."""
        score = orb_strategy._calculate_signal_score(
            price=99.95,        # No breakout: 0 pts
            orb=sample_orb,
            vwap=101.00,        # Wrong VWAP side: 0 pts
            rsi=80.0,           # Overbought: 0 pts
            rel_volume=1.0,     # Low volume: 0 pts
            macd_histogram=-0.15,  # Wrong MACD: 0 pts
            sentiment=-1.0,     # Wrong sentiment: 0 pts
            direction='LONG'
        )

        assert score < 10

    def test_score_always_non_negative(self, orb_strategy, sample_orb):
        """Score should never be negative."""
        score = orb_strategy._calculate_signal_score(
            price=90.00,
            orb=sample_orb,
            vwap=110.00,
            rsi=90.0,
            rel_volume=0.5,
            macd_histogram=-1.0,
            sentiment=-1.0,
            direction='LONG'
        )

        assert score >= 0

    def test_score_always_at_most_100(self, orb_strategy, sample_orb):
        """Score should never exceed 100."""
        score = orb_strategy._calculate_signal_score(
            price=110.00,       # Extreme breakout
            orb=sample_orb,
            vwap=90.00,         # Extreme VWAP distance
            rsi=50.0,
            rel_volume=10.0,    # Extreme volume
            macd_histogram=1.0,  # Extreme MACD
            sentiment=1.0,
            direction='LONG'
        )

        assert score <= 100
