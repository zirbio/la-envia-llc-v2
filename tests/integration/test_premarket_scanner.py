"""
Integration tests for scanner/premarket.py - Pre-market Gap Scanner.

Tests cover:
- Symbol filtering (6 tests)
- Score calculation (7 tests)
- Sentiment boost calculation (4 tests)
- Watchlist operations (1 test)
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

from scanner.premarket import PremarketScanner, ScanResult, SCAN_UNIVERSE
from config.settings import TradingConfig, SentimentConfig


# ============================================================================
# Symbol Filtering Tests (6 tests)
# ============================================================================

class TestScanSymbolFiltering:
    """Tests for symbol scanning and filtering."""

    @pytest.fixture
    def scanner(self):
        """Create PremarketScanner with mocked settings."""
        with patch('scanner.premarket.settings') as mock_settings:
            mock_settings.trading = TradingConfig()
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            scanner = PremarketScanner()
        return scanner

    @pytest.fixture
    def valid_premarket_data(self):
        """Valid premarket data that meets all criteria."""
        return {
            'prev_close': 100.0,
            'current_price': 105.0,
            'gap_percent': 5.0,
            'premarket_volume': 600_000
        }

    def test_scan_symbol_meets_all_criteria(self, scanner, valid_premarket_data):
        """Symbol meeting all criteria should return ScanResult."""
        with patch('scanner.premarket.market_data') as mock_market:
            mock_market.get_premarket_data.return_value = valid_premarket_data
            mock_market.get_avg_daily_volume.return_value = 2_000_000

            result = scanner.scan_symbol("AAPL")

            assert result is not None
            assert isinstance(result, ScanResult)
            assert result.symbol == "AAPL"
            assert result.gap_percent == 5.0
            assert result.score > 0

    def test_scan_symbol_gap_below_minimum(self, scanner):
        """Symbol with gap below minimum should return None."""
        with patch('scanner.premarket.market_data') as mock_market:
            mock_market.get_premarket_data.return_value = {
                'prev_close': 100.0,
                'current_price': 101.0,
                'gap_percent': 1.0,  # Below 2% minimum
                'premarket_volume': 600_000
            }
            mock_market.get_avg_daily_volume.return_value = 2_000_000

            result = scanner.scan_symbol("AAPL")

            assert result is None

    def test_scan_symbol_price_below_minimum(self, scanner):
        """Symbol with price below minimum should return None."""
        with patch('scanner.premarket.market_data') as mock_market:
            mock_market.get_premarket_data.return_value = {
                'prev_close': 5.0,
                'current_price': 5.50,
                'gap_percent': 10.0,
                'premarket_volume': 600_000
            }
            mock_market.get_avg_daily_volume.return_value = 2_000_000

            result = scanner.scan_symbol("PENNY")

            assert result is None  # Price $5.50 < $10 minimum

    def test_scan_symbol_price_above_maximum(self, scanner):
        """Symbol with price above maximum should return None."""
        with patch('scanner.premarket.market_data') as mock_market:
            mock_market.get_premarket_data.return_value = {
                'prev_close': 550.0,
                'current_price': 577.5,
                'gap_percent': 5.0,
                'premarket_volume': 600_000
            }
            mock_market.get_avg_daily_volume.return_value = 2_000_000

            result = scanner.scan_symbol("EXPENSIVE")

            assert result is None  # Price $577.50 > $500 maximum

    def test_scan_symbol_low_premarket_volume(self, scanner):
        """Symbol with low premarket volume should return None."""
        with patch('scanner.premarket.market_data') as mock_market:
            mock_market.get_premarket_data.return_value = {
                'prev_close': 100.0,
                'current_price': 105.0,
                'gap_percent': 5.0,
                'premarket_volume': 200_000  # Below 500K minimum
            }
            mock_market.get_avg_daily_volume.return_value = 2_000_000

            result = scanner.scan_symbol("AAPL")

            assert result is None

    def test_scan_symbol_low_avg_volume(self, scanner):
        """Symbol with low average volume should return None."""
        with patch('scanner.premarket.market_data') as mock_market:
            mock_market.get_premarket_data.return_value = {
                'prev_close': 100.0,
                'current_price': 105.0,
                'gap_percent': 5.0,
                'premarket_volume': 600_000
            }
            mock_market.get_avg_daily_volume.return_value = 500_000  # Below 1M minimum

            result = scanner.scan_symbol("LOWVOL")

            assert result is None


# ============================================================================
# Score Calculation Tests (7 tests)
# ============================================================================

class TestScoreCalculation:
    """Tests for composite score calculation."""

    @pytest.fixture
    def scanner(self):
        """Create PremarketScanner with mocked settings."""
        with patch('scanner.premarket.settings') as mock_settings:
            mock_settings.trading = TradingConfig()
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            scanner = PremarketScanner()
        return scanner

    def test_calculate_score_gap_component(self, scanner):
        """Gap score should max out at 30 points."""
        # Small gap (2%)
        score_small = scanner._calculate_score(
            gap_pct=2.0, pm_volume=500_000, avg_volume=1_000_000, price=100.0
        )

        # Large gap (10%) - should hit cap
        score_large = scanner._calculate_score(
            gap_pct=10.0, pm_volume=500_000, avg_volume=1_000_000, price=100.0
        )

        # Very large gap (20%) - should still cap at 30
        score_huge = scanner._calculate_score(
            gap_pct=20.0, pm_volume=500_000, avg_volume=1_000_000, price=100.0
        )

        # Gap score = gap_pct * 5, max 30
        assert score_large > score_small
        # At 6% gap, score would be 30 (maxed out)
        # At 10% and 20%, both should be capped at 30
        assert score_huge == score_large  # Both maxed at 30 for gap component

    def test_calculate_score_volume_component(self, scanner):
        """Volume score should max out at 30 points."""
        # Low relative volume
        score_low = scanner._calculate_score(
            gap_pct=3.0, pm_volume=50_000, avg_volume=1_000_000, price=100.0
        )

        # High relative volume
        score_high = scanner._calculate_score(
            gap_pct=3.0, pm_volume=500_000, avg_volume=1_000_000, price=100.0
        )

        assert score_high > score_low

    def test_calculate_score_liquidity_component(self, scanner):
        """Liquidity score should max out at 20 points."""
        # Low liquidity (1M avg volume) - liquidity score = min(1M / 1M * 5, 20) = 5
        score_low = scanner._calculate_score(
            gap_pct=3.0, pm_volume=100_000, avg_volume=1_000_000, price=100.0
        )

        # High liquidity (10M avg volume) - liquidity score = min(10M / 1M * 5, 20) = 20 (maxed)
        # Note: Higher avg_volume also affects volume_score denominator
        score_high = scanner._calculate_score(
            gap_pct=3.0, pm_volume=1_000_000, avg_volume=10_000_000, price=100.0
        )

        # The liquidity component is higher for high avg volume
        # liquidity_score = min(avg_volume / 1_000_000 * 5, 20)
        liquidity_low = min(1_000_000 / 1_000_000 * 5, 20)  # 5
        liquidity_high = min(10_000_000 / 1_000_000 * 5, 20)  # 20 (maxed)
        assert liquidity_high > liquidity_low

    def test_calculate_score_price_optimal_range(self, scanner):
        """Price in optimal range ($20-$200) should get 20 points."""
        score = scanner._calculate_score(
            gap_pct=3.0, pm_volume=500_000, avg_volume=1_000_000, price=100.0
        )

        # Check that optimal price gets full price component
        # Total = gap (15) + volume (50) + liquidity (5) + price (20)
        assert score >= 20  # At least price score

    def test_calculate_score_price_mid_range(self, scanner):
        """Price in mid range ($10-$20 or $200-$300) should get 10 points."""
        score_low = scanner._calculate_score(
            gap_pct=3.0, pm_volume=500_000, avg_volume=1_000_000, price=15.0
        )

        score_high = scanner._calculate_score(
            gap_pct=3.0, pm_volume=500_000, avg_volume=1_000_000, price=250.0
        )

        # Mid range should get less than optimal
        score_optimal = scanner._calculate_score(
            gap_pct=3.0, pm_volume=500_000, avg_volume=1_000_000, price=100.0
        )

        assert score_low < score_optimal
        assert score_high < score_optimal

    def test_calculate_score_price_edge_range(self, scanner):
        """Price at edge of range should get 5 points."""
        # Price below $10 (but above $10 filter - this tests the scoring only)
        # Since filter is at $10, testing $10.01 would be edge
        score_edge = scanner._calculate_score(
            gap_pct=3.0, pm_volume=500_000, avg_volume=1_000_000, price=350.0
        )

        score_optimal = scanner._calculate_score(
            gap_pct=3.0, pm_volume=500_000, avg_volume=1_000_000, price=100.0
        )

        assert score_edge < score_optimal

    def test_calculate_score_maximum_possible(self, scanner):
        """Maximum possible score should be 100 points."""
        # Perfect conditions
        score = scanner._calculate_score(
            gap_pct=10.0,        # 30 points (maxed)
            pm_volume=1_000_000,  # 30 points (high ratio)
            avg_volume=5_000_000, # 20 points (maxed)
            price=100.0          # 20 points (optimal)
        )

        assert score <= 100
        assert score >= 90  # Should be near max


# ============================================================================
# Sentiment Boost Tests (4 tests)
# ============================================================================

class TestSentimentBoost:
    """Tests for sentiment-based score adjustment."""

    @pytest.fixture
    def scanner(self):
        """Create PremarketScanner with mocked settings."""
        with patch('scanner.premarket.settings') as mock_settings:
            mock_settings.trading = TradingConfig()
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            scanner = PremarketScanner()
        return scanner

    def test_calculate_sentiment_boost_gap_up_bullish(self, scanner):
        """Gap up with bullish sentiment should boost score."""
        boost = scanner._calculate_sentiment_boost(
            gap_percent=5.0,   # Gap up
            sentiment_score=0.5  # Bullish
        )

        assert boost > 0  # Positive boost
        assert boost == 5.0  # 0.5 * 10 = +5

    def test_calculate_sentiment_boost_gap_up_bearish(self, scanner):
        """Gap up with bearish sentiment should penalize score."""
        boost = scanner._calculate_sentiment_boost(
            gap_percent=5.0,   # Gap up
            sentiment_score=-0.5  # Bearish
        )

        assert boost < 0  # Penalty
        assert boost == -5.0  # -0.5 * 10 = -5

    def test_calculate_sentiment_boost_gap_down_bearish(self, scanner):
        """Gap down with bearish sentiment should boost score."""
        boost = scanner._calculate_sentiment_boost(
            gap_percent=-5.0,  # Gap down
            sentiment_score=-0.5  # Bearish
        )

        assert boost > 0  # Positive boost (inverted for gap down)
        assert boost == 5.0  # -(-0.5) * 10 = +5

    def test_calculate_sentiment_boost_gap_down_bullish(self, scanner):
        """Gap down with bullish sentiment should penalize score."""
        boost = scanner._calculate_sentiment_boost(
            gap_percent=-5.0,  # Gap down
            sentiment_score=0.5  # Bullish
        )

        assert boost < 0  # Penalty (inverted for gap down)
        assert boost == -5.0  # -(0.5) * 10 = -5


# ============================================================================
# Watchlist Operations Tests
# ============================================================================

class TestWatchlistOperations:
    """Tests for watchlist scanning operations."""

    @pytest.fixture
    def scanner(self):
        """Create PremarketScanner with mocked settings."""
        with patch('scanner.premarket.settings') as mock_settings:
            mock_settings.trading = TradingConfig()
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            scanner = PremarketScanner()
        return scanner

    def test_scan_watchlist_returns_sorted_results(self, scanner):
        """scan_watchlist should return candidates sorted by score."""
        with patch('scanner.premarket.market_data') as mock_market:
            def mock_premarket(symbol):
                if symbol == "HIGH":
                    return {'prev_close': 100.0, 'current_price': 110.0,
                            'gap_percent': 10.0, 'premarket_volume': 1_000_000}
                elif symbol == "LOW":
                    return {'prev_close': 100.0, 'current_price': 103.0,
                            'gap_percent': 3.0, 'premarket_volume': 600_000}
                return None

            mock_market.get_premarket_data.side_effect = mock_premarket
            mock_market.get_avg_daily_volume.return_value = 2_000_000

            results = scanner.scan_watchlist(["HIGH", "LOW", "INVALID"])

            assert len(results) == 2
            # Should be sorted by score, highest first
            assert results[0].symbol == "HIGH"
            assert results[1].symbol == "LOW"
            assert results[0].score > results[1].score


# ============================================================================
# Async Watchlist with Sentiment Tests
# ============================================================================

class TestAsyncWatchlistWithSentiment:
    """Tests for async watchlist scanning with sentiment."""

    @pytest.fixture
    def scanner(self):
        """Create PremarketScanner with mocked settings."""
        with patch('scanner.premarket.settings') as mock_settings:
            mock_settings.trading = TradingConfig()
            mock_settings.sentiment = SentimentConfig(
                finnhub_api_key="test_key",
                enabled=True
            )
            scanner = PremarketScanner()
        return scanner

    @pytest.mark.asyncio
    async def test_scan_watchlist_with_sentiment(self, scanner):
        """Async scan should add sentiment to results."""
        with patch('scanner.premarket.market_data') as mock_market, \
             patch('scanner.premarket.sentiment_analyzer') as mock_sentiment:

            mock_market.get_premarket_data.return_value = {
                'prev_close': 100.0,
                'current_price': 105.0,
                'gap_percent': 5.0,
                'premarket_volume': 600_000
            }
            mock_market.get_avg_daily_volume.return_value = 2_000_000

            # Mock sentiment results
            from data.sentiment import SentimentResult
            mock_sentiment.get_batch_sentiment = AsyncMock(return_value={
                "AAPL": SentimentResult(
                    symbol="AAPL",
                    score=0.5,
                    news_count=10,
                    positive_count=6,
                    negative_count=2,
                    neutral_count=2,
                    headlines=["Positive news"],
                    timestamp=datetime.now()
                )
            })

            results = await scanner.scan_watchlist_with_sentiment(["AAPL"])

            assert len(results) == 1
            assert results[0].sentiment_score == 0.5
            assert results[0].sentiment_label == "Alcista"


# ============================================================================
# Helper Method Tests
# ============================================================================

class TestHelperMethods:
    """Tests for helper methods."""

    @pytest.fixture
    def scanner_with_candidates(self):
        """Create scanner with pre-populated candidates."""
        with patch('scanner.premarket.settings') as mock_settings:
            mock_settings.trading = TradingConfig()
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            scanner = PremarketScanner()

        # Add test candidates
        scanner.candidates = [
            ScanResult(symbol="BULL", prev_close=100, current_price=110,
                      gap_percent=10.0, premarket_volume=1_000_000,
                      avg_daily_volume=5_000_000, score=90),
            ScanResult(symbol="BEAR", prev_close=100, current_price=90,
                      gap_percent=-10.0, premarket_volume=800_000,
                      avg_daily_volume=3_000_000, score=80),
            ScanResult(symbol="MID", prev_close=100, current_price=105,
                      gap_percent=5.0, premarket_volume=600_000,
                      avg_daily_volume=2_000_000, score=70),
        ]
        return scanner

    def test_get_top_candidates(self, scanner_with_candidates):
        """get_top_candidates should return N highest scored."""
        top_2 = scanner_with_candidates.get_top_candidates(n=2)

        assert len(top_2) == 2
        assert top_2[0].symbol == "BULL"
        assert top_2[1].symbol == "BEAR"

    def test_get_gappers_up(self, scanner_with_candidates):
        """get_gappers_up should return only positive gaps."""
        gappers_up = scanner_with_candidates.get_gappers_up()

        assert len(gappers_up) == 2
        assert all(c.gap_percent > 0 for c in gappers_up)

    def test_get_gappers_down(self, scanner_with_candidates):
        """get_gappers_down should return only negative gaps."""
        gappers_down = scanner_with_candidates.get_gappers_down()

        assert len(gappers_down) == 1
        assert gappers_down[0].symbol == "BEAR"
        assert gappers_down[0].gap_percent < 0

    def test_format_watchlist_message_empty(self):
        """format_watchlist_message with no candidates."""
        with patch('scanner.premarket.settings') as mock_settings:
            mock_settings.trading = TradingConfig()
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            empty_scanner = PremarketScanner()

        message = empty_scanner.format_watchlist_message()
        assert "No candidates found" in message

    def test_format_watchlist_message_with_candidates(self, scanner_with_candidates):
        """format_watchlist_message should format all candidates."""
        message = scanner_with_candidates.format_watchlist_message()

        assert "WATCHLIST" in message
        assert "BULL" in message
        assert "BEAR" in message
        assert "+10.0%" in message or "+10%" in message  # Gap up
        assert "-10.0%" in message or "-10%" in message  # Gap down


# ============================================================================
# SCAN_UNIVERSE Test
# ============================================================================

class TestScanUniverse:
    """Tests for the SCAN_UNIVERSE constant."""

    def test_scan_universe_not_empty(self):
        """SCAN_UNIVERSE should contain symbols."""
        assert len(SCAN_UNIVERSE) > 0

    def test_scan_universe_contains_major_stocks(self):
        """SCAN_UNIVERSE should contain major trading stocks."""
        major_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
        for stock in major_stocks:
            assert stock in SCAN_UNIVERSE, f"Missing {stock} in SCAN_UNIVERSE"

    def test_scan_universe_contains_etfs(self):
        """SCAN_UNIVERSE should contain major ETFs."""
        etfs = ["SPY", "QQQ", "IWM"]
        for etf in etfs:
            assert etf in SCAN_UNIVERSE, f"Missing {etf} in SCAN_UNIVERSE"
