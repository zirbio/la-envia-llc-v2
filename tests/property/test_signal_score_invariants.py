"""
Property-based tests for signal score invariants using Hypothesis.

These tests verify that _calculate_signal_score() maintains invariants
across a wide range of inputs:
1. Score is always between 0 and 100
2. Higher volume never reduces score (monotonicity)
3. Score is deterministic (same inputs = same output)

Run with: pytest tests/property/test_signal_score_invariants.py -v
"""
import pytest
from datetime import datetime
from unittest.mock import patch
from hypothesis import given, strategies as st, assume, settings

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


def create_orb(high: float, low: float) -> OpeningRange:
    """Helper to create OpeningRange with valid values."""
    return OpeningRange(
        symbol="TEST",
        high=high,
        low=low,
        range_size=high - low,
        vwap=(high + low) / 2,
        timestamp=datetime.now()
    )


# ============================================================================
# Property 1: Score Always 0-100
# ============================================================================

@pytest.mark.property
class TestScoreBounds:
    """Property tests for score bounds invariant."""

    @given(
        price=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        orb_high=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        orb_low=st.floats(min_value=5.0, max_value=400.0, allow_nan=False, allow_infinity=False),
        vwap=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        rsi=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        rel_volume=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        macd_histogram=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        sentiment=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_score_always_between_0_and_100(
        self, price, orb_high, orb_low, vwap, rsi, rel_volume, macd_histogram, sentiment
    ):
        """Score should always be in [0, 100] range."""
        # Ensure orb_high > orb_low
        assume(orb_high > orb_low)
        assume(orb_high - orb_low >= 0.1)

        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.MODERATE
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        orb = create_orb(orb_high, orb_low)

        for direction in ['LONG', 'SHORT']:
            score = strategy._calculate_signal_score(
                price=price,
                orb=orb,
                vwap=vwap,
                rsi=rsi,
                rel_volume=rel_volume,
                macd_histogram=macd_histogram,
                sentiment=sentiment,
                direction=direction
            )

            assert 0 <= score <= 100, f"Score {score} out of bounds for direction {direction}"

    @given(
        price=st.floats(min_value=90.0, max_value=110.0, allow_nan=False, allow_infinity=False),
        vwap=st.floats(min_value=90.0, max_value=110.0, allow_nan=False, allow_infinity=False),
        rsi=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        rel_volume=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        macd_histogram=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
        sentiment=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_score_bounds_realistic_range(
        self, price, vwap, rsi, rel_volume, macd_histogram, sentiment
    ):
        """Score bounds with realistic market values."""
        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.MODERATE
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        orb = create_orb(100.0, 98.0)

        score = strategy._calculate_signal_score(
            price=price,
            orb=orb,
            vwap=vwap,
            rsi=rsi,
            rel_volume=rel_volume,
            macd_histogram=macd_histogram,
            sentiment=sentiment,
            direction='LONG'
        )

        assert 0 <= score <= 100


# ============================================================================
# Property 2: Volume Monotonicity
# ============================================================================

@pytest.mark.property
class TestVolumeMonotonicity:
    """Property tests for volume score monotonicity."""

    @given(
        vol1=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
        vol2=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_higher_volume_never_reduces_score(self, vol1, vol2):
        """Higher volume should never result in lower score."""
        assume(vol2 > vol1)

        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.MODERATE
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        orb = create_orb(100.0, 98.0)

        score1 = strategy._calculate_signal_score(
            price=100.50,
            orb=orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=vol1,
            macd_histogram=0.1,
            sentiment=0.0,
            direction='LONG'
        )

        score2 = strategy._calculate_signal_score(
            price=100.50,
            orb=orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=vol2,
            macd_histogram=0.1,
            sentiment=0.0,
            direction='LONG'
        )

        assert score2 >= score1, f"Higher volume {vol2} gave lower score {score2} vs {score1}"


# ============================================================================
# Property 3: Determinism
# ============================================================================

@pytest.mark.property
class TestScoreDeterminism:
    """Property tests for score determinism."""

    @given(
        price=st.floats(min_value=90.0, max_value=110.0, allow_nan=False, allow_infinity=False),
        vwap=st.floats(min_value=90.0, max_value=110.0, allow_nan=False, allow_infinity=False),
        rsi=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        rel_volume=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
        macd_histogram=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
        sentiment=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_same_inputs_same_output(
        self, price, vwap, rsi, rel_volume, macd_histogram, sentiment
    ):
        """Same inputs should always produce same score."""
        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.MODERATE
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        orb = create_orb(100.0, 98.0)

        kwargs = dict(
            price=price,
            orb=orb,
            vwap=vwap,
            rsi=rsi,
            rel_volume=rel_volume,
            macd_histogram=macd_histogram,
            sentiment=sentiment,
            direction='LONG'
        )

        score1 = strategy._calculate_signal_score(**kwargs)
        score2 = strategy._calculate_signal_score(**kwargs)
        score3 = strategy._calculate_signal_score(**kwargs)

        assert score1 == score2 == score3


# ============================================================================
# Property 4: Component Independence
# ============================================================================

@pytest.mark.property
class TestComponentIndependence:
    """Property tests for score component independence."""

    @given(
        rsi1=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        rsi2=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_rsi_change_only_affects_rsi_component(self, rsi1, rsi2):
        """Changing only RSI should change score by at most 15 points."""
        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.MODERATE
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        orb = create_orb(100.0, 98.0)

        score1 = strategy._calculate_signal_score(
            price=100.50,
            orb=orb,
            vwap=99.00,
            rsi=rsi1,
            rel_volume=2.0,
            macd_histogram=0.1,
            sentiment=0.0,
            direction='LONG'
        )

        score2 = strategy._calculate_signal_score(
            price=100.50,
            orb=orb,
            vwap=99.00,
            rsi=rsi2,
            rel_volume=2.0,
            macd_histogram=0.1,
            sentiment=0.0,
            direction='LONG'
        )

        # RSI component is max 15 points
        assert abs(score1 - score2) <= 15

    @given(
        sentiment1=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        sentiment2=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_sentiment_change_only_affects_sentiment_component(self, sentiment1, sentiment2):
        """Changing only sentiment should change score by at most 10 points."""
        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.MODERATE
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        orb = create_orb(100.0, 98.0)

        score1 = strategy._calculate_signal_score(
            price=100.50,
            orb=orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=2.0,
            macd_histogram=0.1,
            sentiment=sentiment1,
            direction='LONG'
        )

        score2 = strategy._calculate_signal_score(
            price=100.50,
            orb=orb,
            vwap=99.00,
            rsi=50.0,
            rel_volume=2.0,
            macd_histogram=0.1,
            sentiment=sentiment2,
            direction='LONG'
        )

        # Sentiment component is max 10 points
        assert abs(score1 - score2) <= 10


# ============================================================================
# Property 5: Direction Symmetry
# ============================================================================

@pytest.mark.property
class TestDirectionSymmetry:
    """Property tests for LONG/SHORT direction symmetry."""

    @given(
        breakout_pct=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
        vwap_dist=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
        rsi=st.floats(min_value=30.0, max_value=70.0, allow_nan=False, allow_infinity=False),
        rel_volume=st.floats(min_value=1.0, max_value=3.0, allow_nan=False, allow_infinity=False),
        macd_strength=st.floats(min_value=0.05, max_value=0.3, allow_nan=False, allow_infinity=False),
        sentiment_strength=st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_symmetric_conditions_similar_scores(
        self, breakout_pct, vwap_dist, rsi, rel_volume, macd_strength, sentiment_strength
    ):
        """Symmetric LONG and SHORT conditions should produce similar scores."""
        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.MODERATE
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        orb = create_orb(100.0, 98.0)

        # LONG scenario
        long_price = 100.0 * (1 + breakout_pct / 100)
        long_vwap = long_price - vwap_dist
        long_score = strategy._calculate_signal_score(
            price=long_price,
            orb=orb,
            vwap=long_vwap,
            rsi=rsi,
            rel_volume=rel_volume,
            macd_histogram=macd_strength,
            sentiment=sentiment_strength,
            direction='LONG'
        )

        # SHORT scenario (symmetric)
        short_price = 98.0 * (1 - breakout_pct / 100)
        short_vwap = short_price + vwap_dist
        short_score = strategy._calculate_signal_score(
            price=short_price,
            orb=orb,
            vwap=short_vwap,
            rsi=100 - rsi,  # Mirror RSI
            rel_volume=rel_volume,
            macd_histogram=-macd_strength,
            sentiment=-sentiment_strength,
            direction='SHORT'
        )

        # Scores should be roughly similar (within 20 points due to RSI asymmetry)
        assert abs(long_score - short_score) <= 25
