"""
Unit tests for config/settings.py - Configuration settings module.

Tests cover:
- AlpacaConfig (2 tests)
- TradingConfig (2 tests)
- SentimentConfig (1 test)
- Settings.load() (1 test)
"""
import pytest
import os
from unittest.mock import patch

from config.settings import (
    AlpacaConfig,
    TelegramConfig,
    SentimentConfig,
    TradingConfig,
    Settings
)


# ============================================================================
# AlpacaConfig Tests
# ============================================================================

class TestAlpacaConfig:
    """Tests for AlpacaConfig dataclass."""

    def test_alpaca_config_paper_base_url(self):
        """Paper trading should use paper-api URL."""
        config = AlpacaConfig(api_key="test", secret_key="test", paper=True)
        assert config.base_url == "https://paper-api.alpaca.markets"

    def test_alpaca_config_live_base_url(self):
        """Live trading should use production API URL."""
        config = AlpacaConfig(api_key="test", secret_key="test", paper=False)
        assert config.base_url == "https://api.alpaca.markets"


# ============================================================================
# TradingConfig Tests
# ============================================================================

class TestTradingConfig:
    """Tests for TradingConfig dataclass."""

    def test_trading_config_default_values(self):
        """TradingConfig should have correct default values."""
        config = TradingConfig()

        # Capital and risk
        assert config.max_capital == 25000.0
        assert config.risk_per_trade == 0.02
        assert config.max_trades_per_day == 3
        assert config.reward_risk_ratio == 2.0

        # Trading hours
        assert config.market_open == "09:30"
        assert config.orb_end == "09:45"
        assert config.trading_end == "16:00"

        # ORB settings
        assert config.orb_period_minutes == 15

        # Scanner filters
        assert config.min_gap_percent == 2.0
        assert config.min_premarket_volume == 500_000
        assert config.min_price == 10.0
        assert config.max_price == 500.0
        assert config.min_avg_volume == 1_000_000

        # Signal filters
        assert config.min_relative_volume == 1.5
        assert config.rsi_overbought == 70
        assert config.rsi_oversold == 30

    def test_trading_config_custom_values(self):
        """TradingConfig should accept custom values."""
        config = TradingConfig(
            max_capital=50000.0,
            risk_per_trade=0.01,
            max_trades_per_day=5
        )

        assert config.max_capital == 50000.0
        assert config.risk_per_trade == 0.01
        assert config.max_trades_per_day == 5


# ============================================================================
# SentimentConfig Tests
# ============================================================================

class TestSentimentConfig:
    """Tests for SentimentConfig dataclass."""

    def test_sentiment_config_default_values(self):
        """SentimentConfig should have correct default values."""
        config = SentimentConfig(finnhub_api_key="", enabled=True)

        assert config.min_score_long == -0.3
        assert config.max_score_short == 0.3
        assert config.cache_minutes == 30


# ============================================================================
# Settings Integration Test
# ============================================================================

class TestSettings:
    """Tests for Settings class."""

    def test_settings_load_creates_all_configs(self):
        """Settings.load() should create all configuration objects."""
        with patch.dict(os.environ, {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret',
            'ALPACA_PAPER': 'true',
            'TELEGRAM_BOT_TOKEN': 'bot_token',
            'TELEGRAM_CHAT_ID': '12345',
            'FINNHUB_API_KEY': 'finnhub_key',
            'SENTIMENT_ENABLED': 'true'
        }):
            settings = Settings.load()

            # Check all config objects are created
            assert settings.alpaca is not None
            assert settings.telegram is not None
            assert settings.trading is not None
            assert settings.sentiment is not None

            # Check types
            assert isinstance(settings.alpaca, AlpacaConfig)
            assert isinstance(settings.telegram, TelegramConfig)
            assert isinstance(settings.trading, TradingConfig)
            assert isinstance(settings.sentiment, SentimentConfig)
