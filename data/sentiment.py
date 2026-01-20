"""
Sentiment Analysis Module for Trading Signals

Fetches news sentiment from multiple sources to filter trades.
Uses Finnhub API (free tier) and keyword-based analysis.
"""
import asyncio
import re
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
import aiohttp
from loguru import logger


@dataclass
class SentimentResult:
    """Result of sentiment analysis for a symbol"""
    symbol: str
    score: float  # -1.0 (very bearish) to 1.0 (very bullish)
    news_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    headlines: list[str]
    timestamp: datetime


# Sentiment keywords for basic analysis
BULLISH_KEYWORDS = [
    'surge', 'soar', 'rally', 'gain', 'jump', 'rise', 'climb', 'advance',
    'breakout', 'bullish', 'upgrade', 'beat', 'exceeds', 'outperform',
    'growth', 'record', 'high', 'strong', 'profit', 'boom', 'buy',
    'positive', 'momentum', 'optimistic', 'upside', 'recovery', 'win'
]

BEARISH_KEYWORDS = [
    'crash', 'plunge', 'drop', 'fall', 'decline', 'sink', 'tumble', 'slide',
    'selloff', 'bearish', 'downgrade', 'miss', 'below', 'underperform',
    'loss', 'weak', 'low', 'struggle', 'concern', 'risk', 'sell',
    'negative', 'warning', 'pessimistic', 'downside', 'cut', 'fail'
]


class SentimentAnalyzer:
    """
    Analyzes news sentiment for stocks using multiple sources.

    Primary: Finnhub API (free tier: 60 calls/minute)
    Fallback: Keyword-based analysis
    """

    def __init__(self, finnhub_api_key: Optional[str] = None):
        """
        Initialize sentiment analyzer.

        Args:
            finnhub_api_key: Optional Finnhub API key for news data
        """
        self.finnhub_api_key = finnhub_api_key
        self.cache: dict[str, SentimentResult] = {}
        self.cache_ttl = timedelta(minutes=30)  # Cache results for 30 min

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 1.0  # 1 second between requests

    async def get_sentiment(self, symbol: str) -> SentimentResult:
        """
        Get sentiment score for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            SentimentResult with score and details
        """
        # Check cache first
        if symbol in self.cache:
            cached = self.cache[symbol]
            if datetime.now() - cached.timestamp < self.cache_ttl:
                logger.debug(f"Using cached sentiment for {symbol}: {cached.score:.2f}")
                return cached

        # Try Finnhub API
        result = None
        if self.finnhub_api_key:
            result = await self._fetch_finnhub_sentiment(symbol)

        # Fallback to Alpaca news if Finnhub fails
        if result is None:
            result = await self._analyze_with_keywords(symbol)

        # Cache the result
        self.cache[symbol] = result

        return result

    async def get_batch_sentiment(self, symbols: list[str]) -> dict[str, SentimentResult]:
        """
        Get sentiment for multiple symbols efficiently.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict mapping symbol to SentimentResult
        """
        results = {}

        for symbol in symbols:
            try:
                result = await self.get_sentiment(symbol)
                results[symbol] = result
                await asyncio.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.error(f"Error getting sentiment for {symbol}: {e}")
                # Return neutral sentiment on error
                results[symbol] = SentimentResult(
                    symbol=symbol,
                    score=0.0,
                    news_count=0,
                    positive_count=0,
                    negative_count=0,
                    neutral_count=0,
                    headlines=[],
                    timestamp=datetime.now()
                )

        return results

    async def _fetch_finnhub_sentiment(self, symbol: str) -> Optional[SentimentResult]:
        """
        Fetch sentiment from Finnhub API.

        Args:
            symbol: Stock symbol

        Returns:
            SentimentResult or None if failed
        """
        try:
            # Get news from last 24 hours
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)

            url = (
                f"https://finnhub.io/api/v1/company-news"
                f"?symbol={symbol}"
                f"&from={start_date.strftime('%Y-%m-%d')}"
                f"&to={end_date.strftime('%Y-%m-%d')}"
                f"&token={self.finnhub_api_key}"
            )

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        logger.warning(f"Finnhub API error: {response.status}")
                        return None

                    news = await response.json()

            if not news:
                logger.debug(f"No news found for {symbol}")
                return SentimentResult(
                    symbol=symbol,
                    score=0.0,
                    news_count=0,
                    positive_count=0,
                    negative_count=0,
                    neutral_count=0,
                    headlines=[],
                    timestamp=datetime.now()
                )

            # Analyze headlines
            return self._analyze_headlines(symbol, news)

        except asyncio.TimeoutError:
            logger.warning(f"Finnhub request timeout for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error fetching Finnhub data for {symbol}: {e}")
            return None

    async def _analyze_with_keywords(self, symbol: str) -> SentimentResult:
        """
        Fallback: Return neutral sentiment when no API available.

        Args:
            symbol: Stock symbol

        Returns:
            SentimentResult with neutral score
        """
        logger.debug(f"Using neutral sentiment for {symbol} (no API key)")

        return SentimentResult(
            symbol=symbol,
            score=0.0,  # Neutral
            news_count=0,
            positive_count=0,
            negative_count=0,
            neutral_count=0,
            headlines=[],
            timestamp=datetime.now()
        )

    def _analyze_headlines(self, symbol: str, news: list[dict]) -> SentimentResult:
        """
        Analyze news headlines for sentiment.

        Args:
            symbol: Stock symbol
            news: List of news articles from API

        Returns:
            SentimentResult with aggregated score
        """
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        headlines = []

        for article in news[:20]:  # Analyze up to 20 recent articles
            headline = article.get('headline', '')
            summary = article.get('summary', '')

            # Combine headline and summary for analysis
            text = f"{headline} {summary}".lower()
            headlines.append(headline)

            # Count keyword matches
            bullish_matches = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
            bearish_matches = sum(1 for kw in BEARISH_KEYWORDS if kw in text)

            # Classify the article
            if bullish_matches > bearish_matches:
                positive_count += 1
            elif bearish_matches > bullish_matches:
                negative_count += 1
            else:
                neutral_count += 1

        total = positive_count + negative_count + neutral_count

        if total == 0:
            score = 0.0
        else:
            # Calculate weighted score
            # Positive = +1, Neutral = 0, Negative = -1
            score = (positive_count - negative_count) / total

        logger.info(
            f"{symbol} sentiment: {score:.2f} "
            f"(+{positive_count}/-{negative_count}/={neutral_count})"
        )

        return SentimentResult(
            symbol=symbol,
            score=score,
            news_count=total,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            headlines=headlines[:5],  # Keep top 5 headlines
            timestamp=datetime.now()
        )

    def analyze_text(self, text: str) -> float:
        """
        Analyze arbitrary text for sentiment.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score from -1.0 to 1.0
        """
        text_lower = text.lower()

        bullish_matches = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
        bearish_matches = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)

        total = bullish_matches + bearish_matches

        if total == 0:
            return 0.0

        return (bullish_matches - bearish_matches) / total

    def is_sentiment_favorable(
        self,
        score: float,
        direction: str,
        threshold: float = 0.3
    ) -> bool:
        """
        Check if sentiment is favorable for a trade direction.

        Args:
            score: Sentiment score (-1.0 to 1.0)
            direction: Trade direction ('long' or 'short')
            threshold: Minimum absolute score to consider unfavorable

        Returns:
            True if sentiment supports the trade direction
        """
        if direction.lower() == 'long':
            # For long trades, avoid strongly negative sentiment
            return score >= -threshold
        else:
            # For short trades, avoid strongly positive sentiment
            return score <= threshold

    def clear_cache(self):
        """Clear the sentiment cache"""
        self.cache.clear()
        logger.info("Sentiment cache cleared")

    def format_sentiment_message(self, result: SentimentResult) -> str:
        """
        Format sentiment result for Telegram message.

        Args:
            result: SentimentResult to format

        Returns:
            Formatted string for display
        """
        if result.score >= 0.3:
            emoji = "🟢"
            label = "Alcista"
        elif result.score <= -0.3:
            emoji = "🔴"
            label = "Bajista"
        else:
            emoji = "🟡"
            label = "Neutral"

        message = (
            f"{emoji} *{result.symbol} Sentimiento: {label}*\n"
            f"Score: {result.score:.2f}\n"
            f"Noticias: {result.news_count} artículos\n"
            f"(+{result.positive_count}/-{result.negative_count}/={result.neutral_count})"
        )

        if result.headlines:
            message += "\n\n*Titulares recientes:*\n"
            for headline in result.headlines[:3]:
                # Truncate long headlines
                if len(headline) > 60:
                    headline = headline[:57] + "..."
                message += f"• {headline}\n"

        return message


# Global instance
sentiment_analyzer = SentimentAnalyzer()
