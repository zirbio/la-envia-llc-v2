"""
Custom reporting and metrics for ORB backtests
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class BacktestMetrics:
    """Container for backtest performance metrics"""
    total_return: float
    total_return_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_win: float
    avg_loss: float
    avg_win_pct: float
    avg_loss_pct: float
    expectancy: float
    expectancy_pct: float
    best_trade: float
    worst_trade: float
    avg_trade_duration: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "avg_win_pct": self.avg_win_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "expectancy": self.expectancy,
            "expectancy_pct": self.expectancy_pct,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "avg_trade_duration": self.avg_trade_duration,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
        }


class BacktestReporter:
    """Generate reports and analyze backtest results"""

    def __init__(self, trades: list[dict], equity_curve: pd.Series, initial_capital: float):
        """
        Initialize reporter with backtest results

        Args:
            trades: List of trade dictionaries
            equity_curve: Series of portfolio values over time
            initial_capital: Starting capital
        """
        self.trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital

    def calculate_metrics(self) -> BacktestMetrics:
        """Calculate all performance metrics"""
        if self.trades.empty:
            return self._empty_metrics()

        # Basic trade stats
        total_trades = len(self.trades)
        winning_trades = len(self.trades[self.trades["pnl"] > 0])
        losing_trades = len(self.trades[self.trades["pnl"] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # PnL calculations
        total_return = self.trades["pnl"].sum()
        gross_profit = self.trades[self.trades["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(self.trades[self.trades["pnl"] < 0]["pnl"].sum())

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average win/loss
        avg_win = self.trades[self.trades["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0
        avg_loss = abs(self.trades[self.trades["pnl"] < 0]["pnl"].mean()) if losing_trades > 0 else 0

        avg_win_pct = self.trades[self.trades["pnl"] > 0]["pnl_pct"].mean() * 100 if winning_trades > 0 else 0
        avg_loss_pct = abs(self.trades[self.trades["pnl"] < 0]["pnl_pct"].mean()) * 100 if losing_trades > 0 else 0

        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        expectancy_pct = (win_rate * avg_win_pct) - ((1 - win_rate) * avg_loss_pct)

        # Best/worst trades
        best_trade = self.trades["pnl"].max()
        worst_trade = self.trades["pnl"].min()

        # Drawdown
        max_dd, max_dd_pct = self._calculate_max_drawdown()

        # Sharpe ratio
        sharpe = self._calculate_sharpe_ratio()

        # Final values
        final_capital = self.equity_curve.iloc[-1] if not self.equity_curve.empty else self.initial_capital
        total_return_pct = ((final_capital - self.initial_capital) / self.initial_capital) * 100

        # Dates
        start_date = str(self.equity_curve.index[0].date()) if not self.equity_curve.empty else "N/A"
        end_date = str(self.equity_curve.index[-1].date()) if not self.equity_curve.empty else "N/A"

        return BacktestMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            expectancy=expectancy,
            expectancy_pct=expectancy_pct,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_trade_duration="N/A",
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
        )

    def _calculate_max_drawdown(self) -> tuple[float, float]:
        """Calculate maximum drawdown"""
        if self.equity_curve.empty:
            return 0.0, 0.0

        peak = self.equity_curve.expanding(min_periods=1).max()
        drawdown = self.equity_curve - peak
        max_dd = drawdown.min()
        max_dd_pct = (max_dd / peak[drawdown.idxmin()]) * 100 if max_dd < 0 else 0

        return abs(max_dd), abs(max_dd_pct)

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio"""
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0

        returns = self.equity_curve.pct_change().dropna()

        if returns.std() == 0:
            return 0.0

        # Annualize (assuming daily data)
        excess_returns = returns.mean() - (risk_free_rate / 252)
        sharpe = (excess_returns / returns.std()) * np.sqrt(252)

        return sharpe

    def _empty_metrics(self) -> BacktestMetrics:
        """Return empty metrics when no trades"""
        return BacktestMetrics(
            total_return=0,
            total_return_pct=0,
            win_rate=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            profit_factor=0,
            sharpe_ratio=0,
            max_drawdown=0,
            max_drawdown_pct=0,
            avg_win=0,
            avg_loss=0,
            avg_win_pct=0,
            avg_loss_pct=0,
            expectancy=0,
            expectancy_pct=0,
            best_trade=0,
            worst_trade=0,
            avg_trade_duration="N/A",
            start_date="N/A",
            end_date="N/A",
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
        )

    def print_summary(self, metrics: Optional[BacktestMetrics] = None):
        """Print formatted summary to console"""
        if metrics is None:
            metrics = self.calculate_metrics()

        print("\n" + "=" * 50)
        print("ORB Strategy Backtest Results")
        print("=" * 50)
        print(f"Period: {metrics.start_date} to {metrics.end_date}")
        print("-" * 50)
        print(f"Initial Capital:  ${metrics.initial_capital:>15,.2f}")
        print(f"Final Capital:    ${metrics.final_capital:>15,.2f}")
        print(f"Total Return:     ${metrics.total_return:>15,.2f} ({metrics.total_return_pct:+.2f}%)")
        print("-" * 50)
        print(f"Total Trades:     {metrics.total_trades:>15}")
        print(f"Win Rate:         {metrics.win_rate * 100:>14.1f}% ({metrics.winning_trades}/{metrics.total_trades})")
        print(f"Profit Factor:    {metrics.profit_factor:>15.2f}")
        print(f"Sharpe Ratio:     {metrics.sharpe_ratio:>15.2f}")
        print(f"Max Drawdown:     ${metrics.max_drawdown:>14,.2f} ({metrics.max_drawdown_pct:.2f}%)")
        print("-" * 50)
        print(f"Avg Win:          ${metrics.avg_win:>15,.2f} ({metrics.avg_win_pct:+.2f}%)")
        print(f"Avg Loss:         ${metrics.avg_loss:>15,.2f} ({metrics.avg_loss_pct:-.2f}%)")
        print(f"Expectancy:       ${metrics.expectancy:>15,.2f} ({metrics.expectancy_pct:+.2f}%)")
        print("-" * 50)
        print(f"Best Trade:       ${metrics.best_trade:>15,.2f}")
        print(f"Worst Trade:      ${metrics.worst_trade:>15,.2f}")
        print("=" * 50 + "\n")

    def save_json(self, filepath: str, metrics: Optional[BacktestMetrics] = None):
        """Save metrics to JSON file"""
        if metrics is None:
            metrics = self.calculate_metrics()

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        logger.info(f"Metrics saved to {output_path}")
