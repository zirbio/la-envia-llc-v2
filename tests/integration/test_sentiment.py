"""
Integration tests for data/sentiment.py - Sentiment Analysis Module.

Tests cover:
- Caching behavior (3 tests)
- Finnhub API fetching (4 tests)
- Headline analysis (5 tests)
- Batch sentiment (3 tests)
- Utility methods (5 tests)
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

from data.sentiment import (
    SentimentAnalyzer,
    SentimentResult,
    BULLISH_KEYWORDS,
    BEARISH_KEYWORDS
)


# ============================================================================
# Caching Tests (3 tests)
# ============================================================================

class TestSentimentCaching:
    """Tests for sentiment caching behavior."""

    @pytest.fixture
    def analyzer(self):
        """Create SentimentAnalyzer with API key."""
        with patch('data.sentiment.settings') as mock_settings:
            mock_settings.sentiment.cache_minutes = 30
            analyzer = SentimentAnalyzer(finnhub_api_key="test_key")
        return analyzer

    @pytest.mark.asyncio
    async def test_get_sentiment_returns_cached_result(self, analyzer):
        """Cached sentiment should be returned without API call."""
        # Pre-populate cache
        cached_result = SentimentResult(
            symbol="AAPL",
            score=0.5,
            news_count=10,
            positive_count=6,
            negative_count=2,
            neutral_count=2,
            headlines=["Test headline"],
            timestamp=datetime.now()
        )
        analyzer.cache["AAPL"] = cached_result

        # Mock the API method to ensure it's not called
        analyzer._fetch_finnhub_sentiment = AsyncMock()

        result = await analyzer.get_sentiment("AAPL")

        assert result == cached_result
        analyzer._fetch_finnhub_sentiment.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_sentiment_cache_expired(self, analyzer):
        """Expired cache should trigger new API call."""
        # Pre-populate cache with expired entry
        old_result = SentimentResult(
            symbol="AAPL",
            score=0.2,
            news_count=5,
            positive_count=3,
            negative_count=1,
            neutral_count=1,
            headlines=["Old headline"],
            timestamp=datetime.now() - timedelta(hours=1)  # Expired (>30 min)
        )
        analyzer.cache["AAPL"] = old_result

        # Mock API to return new result
        new_result = SentimentResult(
            symbol="AAPL",
            score=0.8,
            news_count=15,
            positive_count=10,
            negative_count=2,
            neutral_count=3,
            headlines=["New headline"],
            timestamp=datetime.now()
        )
        analyzer._fetch_finnhub_sentiment = AsyncMock(return_value=new_result)

        result = await analyzer.get_sentiment("AAPL")

        assert result.score == 0.8
        analyzer._fetch_finnhub_sentiment.assert_called_once()

    def test_clear_cache_empties_dict(self, analyzer):
        """clear_cache should empty the cache dictionary."""
        # Add some entries
        analyzer.cache["AAPL"] = SentimentResult(
            symbol="AAPL", score=0.5, news_count=10,
            positive_count=6, negative_count=2, neutral_count=2,
            headlines=[], timestamp=datetime.now()
        )
        analyzer.cache["TSLA"] = SentimentResult(
            symbol="TSLA", score=0.3, news_count=5,
            positive_count=3, negative_count=1, neutral_count=1,
            headlines=[], timestamp=datetime.now()
        )

        analyzer.clear_cache()

        assert len(analyzer.cache) == 0


# ============================================================================
# Finnhub API Tests (4 tests)
# ============================================================================

class TestFinnhubAPI:
    """Tests for Finnhub API fetching."""

    @pytest.fixture
    def analyzer(self):
        """Create SentimentAnalyzer with API key."""
        with patch('data.sentiment.settings') as mock_settings:
            mock_settings.sentiment.cache_minutes = 30
            analyzer = SentimentAnalyzer(finnhub_api_key="test_key")
        return analyzer

    @pytest.mark.asyncio
    async def test_fetch_finnhub_sentiment_success(self, analyzer):
        """Successful API call should return SentimentResult."""
        mock_news = [
            {"headline": "Stock surges on strong earnings", "summary": "Great results"},
            {"headline": "Company beats expectations", "summary": "Bullish outlook"},
        ]

        # Create a proper async context manager mock
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_news)

        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get_cm)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            result = await analyzer._fetch_finnhub_sentiment("AAPL")

        assert result is not None
        assert isinstance(result, SentimentResult)
        assert result.symbol == "AAPL"
        assert result.news_count > 0

    @pytest.mark.asyncio
    async def test_fetch_finnhub_sentiment_empty_news(self, analyzer):
        """Empty news should return neutral sentiment (score=0.0)."""
        # Create a proper async context manager mock
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[])  # Empty news

        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get_cm)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            result = await analyzer._fetch_finnhub_sentiment("AAPL")

        assert result is not None
        assert result.score == 0.0
        assert result.news_count == 0

    @pytest.mark.asyncio
    async def test_fetch_finnhub_sentiment_api_error(self, analyzer):
        """API error should return None."""
        with patch('aiohttp.ClientSession') as MockSession:
            mock_response = AsyncMock()
            mock_response.status = 500  # Server error

            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response

            MockSession.return_value.__aenter__.return_value = mock_session

            result = await analyzer._fetch_finnhub_sentiment("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_finnhub_sentiment_timeout(self, analyzer):
        """Timeout should return None."""
        with patch('aiohttp.ClientSession') as MockSession:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.side_effect = asyncio.TimeoutError()

            MockSession.return_value.__aenter__.return_value = mock_session

            result = await analyzer._fetch_finnhub_sentiment("AAPL")

        assert result is None


# ============================================================================
# Headline Analysis Tests (5 tests)
# ============================================================================

class TestHeadlineAnalysis:
    """Tests for headline sentiment analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create SentimentAnalyzer."""
        with patch('data.sentiment.settings') as mock_settings:
            mock_settings.sentiment.cache_minutes = 30
            analyzer = SentimentAnalyzer()
        return analyzer

    def test_analyze_headlines_all_bullish(self, analyzer, bullish_news):
        """All bullish headlines should produce score near 1.0."""
        result = analyzer._analyze_headlines("AAPL", bullish_news)

        assert result.score > 0.5
        assert result.positive_count > result.negative_count

    def test_analyze_headlines_all_bearish(self, analyzer, bearish_news):
        """All bearish headlines should produce score near -1.0."""
        result = analyzer._analyze_headlines("AAPL", bearish_news)

        assert result.score < -0.5
        assert result.negative_count > result.positive_count

    def test_analyze_headlines_mixed(self, analyzer):
        """Mixed headlines should produce score near 0."""
        mixed_news = [
            {"headline": "Stock surges on news", "summary": "Bullish momentum"},
            {"headline": "Stock crashes on concerns", "summary": "Risk of decline"},
        ]

        result = analyzer._analyze_headlines("AAPL", mixed_news)

        # Score should be close to 0 (balanced)
        assert abs(result.score) < 0.5

    def test_analyze_headlines_no_keywords(self, analyzer):
        """Headlines with no keywords should produce score=0.0."""
        neutral_news = [
            {"headline": "Company reports results", "summary": "Standard update"},
            {"headline": "CEO makes announcement", "summary": "Business as usual"},
        ]

        result = analyzer._analyze_headlines("AAPL", neutral_news)

        # No keywords matched, all neutral
        assert result.neutral_count == 2
        assert result.score == 0.0

    def test_analyze_headlines_limits_to_20(self, analyzer):
        """Analysis should process maximum 20 articles."""
        # Create 30 articles
        many_articles = [
            {"headline": f"Stock surges {i}", "summary": f"Bullish momentum {i}"}
            for i in range(30)
        ]

        result = analyzer._analyze_headlines("AAPL", many_articles)

        # Should only analyze first 20
        assert result.news_count <= 20


# ============================================================================
# Batch Sentiment Tests (3 tests)
# ============================================================================

class TestBatchSentiment:
    """Tests for batch sentiment retrieval."""

    @pytest.fixture
    def analyzer(self):
        """Create SentimentAnalyzer with API key."""
        with patch('data.sentiment.settings') as mock_settings:
            mock_settings.sentiment.cache_minutes = 30
            analyzer = SentimentAnalyzer(finnhub_api_key="test_key")
        return analyzer

    @pytest.mark.asyncio
    async def test_get_batch_sentiment_multiple_symbols(self, analyzer):
        """Batch request should return results for all symbols."""
        # Mock get_sentiment to return results
        async def mock_get_sentiment(symbol):
            return SentimentResult(
                symbol=symbol,
                score=0.3,
                news_count=5,
                positive_count=3,
                negative_count=1,
                neutral_count=1,
                headlines=[],
                timestamp=datetime.now()
            )

        analyzer.get_sentiment = mock_get_sentiment

        results = await analyzer.get_batch_sentiment(["AAPL", "TSLA", "NVDA"])

        assert len(results) == 3
        assert "AAPL" in results
        assert "TSLA" in results
        assert "NVDA" in results

    @pytest.mark.asyncio
    async def test_get_batch_sentiment_rate_limiting(self, analyzer):
        """Batch requests should have delay between calls (rate limiting)."""
        call_times = []

        async def mock_get_sentiment(symbol):
            call_times.append(datetime.now())
            return SentimentResult(
                symbol=symbol, score=0.0, news_count=0,
                positive_count=0, negative_count=0, neutral_count=0,
                headlines=[], timestamp=datetime.now()
            )

        analyzer.get_sentiment = mock_get_sentiment

        await analyzer.get_batch_sentiment(["AAPL", "TSLA"])

        # Should have 1.0s delay between calls
        if len(call_times) >= 2:
            delta = (call_times[1] - call_times[0]).total_seconds()
            assert delta >= 0.9  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_get_batch_sentiment_error_returns_neutral(self, analyzer):
        """Error for one symbol should return neutral sentiment."""
        call_count = [0]

        async def mock_get_sentiment(symbol):
            call_count[0] += 1
            if symbol == "FAIL":
                raise Exception("API Error")
            return SentimentResult(
                symbol=symbol, score=0.5, news_count=5,
                positive_count=3, negative_count=1, neutral_count=1,
                headlines=[], timestamp=datetime.now()
            )

        analyzer.get_sentiment = mock_get_sentiment

        results = await analyzer.get_batch_sentiment(["AAPL", "FAIL"])

        # AAPL should have result, FAIL should have neutral
        assert results["AAPL"].score == 0.5
        assert results["FAIL"].score == 0.0


# ============================================================================
# Utility Method Tests (5 tests)
# ============================================================================

class TestUtilityMethods:
    """Tests for utility methods."""

    @pytest.fixture
    def analyzer(self):
        """Create SentimentAnalyzer."""
        with patch('data.sentiment.settings') as mock_settings:
            mock_settings.sentiment.cache_minutes = 30
            analyzer = SentimentAnalyzer()
        return analyzer

    def test_analyze_text_bullish(self, analyzer):
        """Text with bullish keywords should return positive score."""
        text = "The stock is showing strong momentum with bullish breakout potential"
        score = analyzer.analyze_text(text)

        assert score > 0

    def test_analyze_text_bearish(self, analyzer):
        """Text with bearish keywords should return negative score."""
        text = "The stock crashed amid concerns about declining sales and risk"
        score = analyzer.analyze_text(text)

        assert score < 0

    def test_analyze_text_neutral(self, analyzer):
        """Text with no keywords should return 0.0."""
        text = "The company held its annual meeting yesterday"
        score = analyzer.analyze_text(text)

        assert score == 0.0

    def test_is_sentiment_favorable_long(self, analyzer):
        """Sentiment favorable check for long trades."""
        # Favorable for long (not strongly negative)
        assert analyzer.is_sentiment_favorable(0.5, 'long') is True
        assert analyzer.is_sentiment_favorable(0.0, 'long') is True
        assert analyzer.is_sentiment_favorable(-0.2, 'long') is True

        # Not favorable for long (too negative)
        assert analyzer.is_sentiment_favorable(-0.5, 'long') is False

    def test_is_sentiment_favorable_short(self, analyzer):
        """Sentiment favorable check for short trades."""
        # Favorable for short (not strongly positive)
        assert analyzer.is_sentiment_favorable(-0.5, 'short') is True
        assert analyzer.is_sentiment_favorable(0.0, 'short') is True
        assert analyzer.is_sentiment_favorable(0.2, 'short') is True

        # Not favorable for short (too positive)
        assert analyzer.is_sentiment_favorable(0.5, 'short') is False

    def test_format_sentiment_message(self, analyzer, sample_sentiment_result):
        """format_sentiment_message should produce formatted string."""
        message = analyzer.format_sentiment_message(sample_sentiment_result)

        assert "AAPL" in message
        assert "Sentimiento" in message
        # Should include stats
        assert str(sample_sentiment_result.news_count) in message


# ============================================================================
# Keyword Lists Tests
# ============================================================================

class TestKeywordLists:
    """Tests for keyword lists."""

    def test_bullish_keywords_not_empty(self):
        """BULLISH_KEYWORDS should contain keywords."""
        assert len(BULLISH_KEYWORDS) > 0

    def test_bearish_keywords_not_empty(self):
        """BEARISH_KEYWORDS should contain keywords."""
        assert len(BEARISH_KEYWORDS) > 0

    def test_no_keyword_overlap(self):
        """Bullish and bearish keywords should not overlap."""
        overlap = set(BULLISH_KEYWORDS) & set(BEARISH_KEYWORDS)
        assert len(overlap) == 0, f"Overlapping keywords: {overlap}"

    def test_bullish_keywords_expected(self):
        """BULLISH_KEYWORDS should contain expected words."""
        expected = ['surge', 'bullish', 'growth', 'strong']
        for word in expected:
            assert word in BULLISH_KEYWORDS

    def test_bearish_keywords_expected(self):
        """BEARISH_KEYWORDS should contain expected words."""
        expected = ['crash', 'bearish', 'decline', 'risk']
        for word in expected:
            assert word in BEARISH_KEYWORDS
