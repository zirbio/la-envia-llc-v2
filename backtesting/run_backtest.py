"""
CLI entrypoint for running ORB strategy backtests
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

from loguru import logger

from lumibot.backtesting import YahooDataBacktesting
from lumibot.traders import Trader

from backtesting.orb_backtest import ORBBacktestStrategy
from backtesting.reports import BacktestReporter
from backtesting.data_loader import BacktestDataLoader
from config.settings import settings

# Try to import Alpaca backtesting (preferred for intraday)
try:
    from lumibot.backtesting import AlpacaBacktesting
    HAS_ALPACA_BACKTEST = True
except ImportError:
    HAS_ALPACA_BACKTEST = False


def get_alpaca_creds(data_feed: str = "iex") -> dict:
    """
    Get Alpaca credentials configuration.

    Args:
        data_feed: Data feed to use ('iex' for free tier, 'sip' for paid)

    Returns:
        Dict with Alpaca credentials for Lumibot
    """
    return {
        "API_KEY": settings.alpaca.api_key,
        "API_SECRET": settings.alpaca.secret_key,
        "PAPER": settings.alpaca.paper,
        "DATA_FEED": data_feed,
    }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run ORB Strategy Backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to backtest (from today)"
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Overrides --days"
    )

    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Defaults to today"
    )

    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY,QQQ,AAPL,TSLA,NVDA",
        help="Comma-separated list of symbols to trade"
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=settings.trading.max_capital,
        help="Initial capital for backtest"
    )

    parser.add_argument(
        "--orb-period",
        type=int,
        default=settings.trading.orb_period_minutes,
        help="Opening Range period in minutes"
    )

    parser.add_argument(
        "--risk",
        type=float,
        default=settings.trading.risk_per_trade,
        help="Risk per trade as decimal (e.g., 0.02 for 2%%)"
    )

    parser.add_argument(
        "--rsi-overbought",
        type=int,
        default=settings.trading.rsi_overbought,
        help="RSI overbought threshold"
    )

    parser.add_argument(
        "--rsi-oversold",
        type=int,
        default=settings.trading.rsi_oversold,
        help="RSI oversold threshold"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory for output reports"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--data-source",
        type=str,
        choices=["yahoo", "alpaca"],
        default="yahoo",
        help="Data source for backtesting. 'alpaca' requires paid subscription for minute data"
    )

    parser.add_argument(
        "--data-feed",
        type=str,
        choices=["iex", "sip"],
        default="iex",
        help="Alpaca data feed. 'iex' is free tier, 'sip' requires paid subscription"
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached data before running backtest"
    )

    parser.add_argument(
        "--cache-max-age",
        type=int,
        default=7,
        help="Maximum age of cache files in days before refresh"
    )

    return parser.parse_args()


def run_backtest(args):
    """Execute the backtest"""
    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Handle cache clearing if requested
    if args.clear_cache:
        logger.info("Clearing data cache...")
        data_loader = BacktestDataLoader(max_cache_age_days=args.cache_max_age)
        cleared = data_loader.clear_cache()
        logger.info(f"Cleared {cleared} cache files")

    # Parse dates
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start_date = datetime.now() - timedelta(days=args.days)

    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        end_date = datetime.now()

    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    logger.info(f"Starting ORB Strategy Backtest")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Initial Capital: ${args.capital:,.2f}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure strategy parameters
    strategy_params = {
        "orb_period_minutes": args.orb_period,
        "risk_per_trade": args.risk,
        "reward_risk_ratio": settings.trading.reward_risk_ratio,
        "min_relative_volume": settings.trading.min_relative_volume,
        "rsi_overbought": args.rsi_overbought,
        "rsi_oversold": args.rsi_oversold,
        "max_trades_per_day": settings.trading.max_trades_per_day,
        "symbols": symbols,
    }

    # Run backtest
    tearsheet_path = str(output_dir / f"backtest_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.html")

    # Select backtesting data source based on user preference
    use_alpaca = (
        args.data_source == "alpaca" and
        HAS_ALPACA_BACKTEST and
        settings.alpaca.api_key
    )

    if use_alpaca:
        logger.info(f"Using Alpaca data source (minute bars, feed: {args.data_feed})")
        if args.data_feed == "sip":
            logger.info("Note: SIP feed requires paid Alpaca subscription")
        else:
            logger.info("Note: IEX feed is free but has 15-min delay for real-time data")

        alpaca_creds = get_alpaca_creds(data_feed=args.data_feed)
        ORBBacktestStrategy.backtest(
            AlpacaBacktesting,
            start_date,
            end_date,
            budget=args.capital,
            parameters=strategy_params,
            show_plot=True,
            show_tearsheet=True,
            save_tearsheet=True,
            tearsheet_file=tearsheet_path,
            config=alpaca_creds,
        )
    else:
        logger.info("Using Yahoo data source (daily bars)")
        logger.info("Note: ORB strategy is simplified for daily data")
        logger.info("For accurate minute-level backtesting, use --data-source alpaca with paid subscription")
        ORBBacktestStrategy.backtest(
            YahooDataBacktesting,
            start_date,
            end_date,
            budget=args.capital,
            parameters=strategy_params,
            show_plot=True,
            show_tearsheet=True,
            save_tearsheet=True,
            tearsheet_file=tearsheet_path,
        )

    logger.info(f"Tearsheet saved to: {tearsheet_path}")

    # Generate custom performance report
    ORBBacktestStrategy.generate_final_report()

    logger.info(f"Backtest complete! Report saved to {output_dir}")


def main():
    """Main entry point"""
    args = parse_args()

    try:
        run_backtest(args)
    except KeyboardInterrupt:
        logger.info("Backtest interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


if __name__ == "__main__":
    main()
