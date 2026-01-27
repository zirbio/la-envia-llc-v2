"""
Positions Screen

Display and monitor open positions.
"""
from typing import List

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from prompt_toolkit import prompt
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

from execution.orders import order_executor, Position


class PositionsScreen:
    """Positions display and monitoring screen"""

    def __init__(self, console: Console = None):
        self.console = console

    def display(self, positions: List[Position] = None):
        """
        Display positions summary.

        Args:
            positions: Optional list of positions (fetches if not provided)
        """
        if positions is None:
            positions = order_executor.get_positions()

        if not RICH_AVAILABLE or self.console is None:
            self._display_basic(positions)
        else:
            self._display_rich(positions)

    def _display_basic(self, positions: List[Position]):
        """Display without Rich"""
        account = order_executor.get_account()

        print("\n" + "=" * 60)
        print("ESTADO DE CUENTA")
        print("=" * 60)

        if account:
            print(f"Equity: ${account.get('equity', 0):,.2f}")
            print(f"Cash: ${account.get('cash', 0):,.2f}")
            print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
            print(f"Day Trades: {account.get('day_trade_count', 0)}")

        print(f"\nPosiciones abiertas: {len(positions)}")
        print("-" * 60)

        if not positions:
            print("Sin posiciones abiertas")
        else:
            total_pnl = 0
            for pos in positions:
                direction = "LONG" if pos.side == 'long' else "SHORT"
                pnl_sign = "+" if pos.unrealized_pl >= 0 else ""
                print(
                    f"{pos.symbol:6s} | {direction:5s} | {pos.qty:4d} @ ${pos.entry_price:>8.2f} | "
                    f"${pos.current_price:>8.2f} | {pnl_sign}${pos.unrealized_pl:>8.2f} "
                    f"({pnl_sign}{pos.unrealized_pl_pct:.1f}%)"
                )
                total_pnl += pos.unrealized_pl

            print("-" * 60)
            total_sign = "+" if total_pnl >= 0 else ""
            print(f"Total P/L: {total_sign}${total_pnl:,.2f}")

    def _display_rich(self, positions: List[Position]):
        """Display with Rich"""
        account = order_executor.get_account()

        # Account info panel
        account_table = Table(show_header=False, box=box.SIMPLE)
        account_table.add_column("Metric", style="dim")
        account_table.add_column("Value", style="bold")

        if account:
            equity = account.get('equity', 0)
            cash = account.get('cash', 0)
            bp = account.get('buying_power', 0)
            dt = account.get('day_trade_count', 0)

            account_table.add_row("Equity", f"${equity:,.2f}")
            account_table.add_row("Cash", f"${cash:,.2f}")
            account_table.add_row("Buying Power", f"${bp:,.2f}")
            account_table.add_row("Day Trades", str(dt))

        self.console.print(Panel(
            account_table,
            title="[bold]CUENTA[/bold]",
            border_style="cyan"
        ))

        # Positions table
        if not positions:
            self.console.print(Panel(
                "Sin posiciones abiertas",
                style="dim",
                border_style="blue"
            ))
            return

        pos_table = Table(box=box.ROUNDED, border_style="blue")
        pos_table.add_column("Symbol", style="bold")
        pos_table.add_column("Side")
        pos_table.add_column("Qty", justify="right")
        pos_table.add_column("Entry", justify="right")
        pos_table.add_column("Current", justify="right")
        pos_table.add_column("P/L", justify="right")
        pos_table.add_column("P/L %", justify="right")

        total_pnl = 0
        for pos in positions:
            pnl_style = "green" if pos.unrealized_pl >= 0 else "red"
            side_style = "cyan" if pos.side == 'long' else "magenta"

            pos_table.add_row(
                pos.symbol,
                pos.side.upper(),
                str(pos.qty),
                f"${pos.entry_price:.2f}",
                f"${pos.current_price:.2f}",
                f"${pos.unrealized_pl:+.2f}",
                f"{pos.unrealized_pl_pct:+.1f}%",
                style=pnl_style
            )
            total_pnl += pos.unrealized_pl

        # Add total row
        total_style = "bold green" if total_pnl >= 0 else "bold red"
        pos_table.add_row("", "", "", "", "[bold]TOTAL[/bold]", f"${total_pnl:+,.2f}", "", style=total_style)

        self.console.print(Panel(
            pos_table,
            title=f"[bold]POSICIONES ({len(positions)})[/bold]",
            border_style="blue"
        ))

    def wait_for_input(self):
        """Wait for user input to continue"""
        if RICH_AVAILABLE and self.console:
            self.console.print()

        if PROMPT_TOOLKIT_AVAILABLE:
            prompt("Presione Enter para continuar...")
        else:
            input("\nPresione Enter para continuar...")


class LivePositionsScreen:
    """Live updating positions display"""

    def __init__(self, console: Console = None):
        self.console = console
        self.running = False

    def _generate_table(self) -> Table:
        """Generate positions table for live display"""
        positions = order_executor.get_positions()
        account = order_executor.get_account()

        # Main table
        table = Table(box=box.ROUNDED, title="Posiciones en Vivo")
        table.add_column("Symbol", style="bold")
        table.add_column("Side")
        table.add_column("Qty", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P/L", justify="right")
        table.add_column("P/L %", justify="right")

        if not positions:
            table.add_row("", "", "", "Sin posiciones", "", "", "")
        else:
            total_pnl = 0
            for pos in positions:
                pnl_style = "green" if pos.unrealized_pl >= 0 else "red"
                table.add_row(
                    pos.symbol,
                    pos.side.upper(),
                    str(pos.qty),
                    f"${pos.entry_price:.2f}",
                    f"${pos.current_price:.2f}",
                    f"${pos.unrealized_pl:+.2f}",
                    f"{pos.unrealized_pl_pct:+.1f}%",
                    style=pnl_style
                )
                total_pnl += pos.unrealized_pl

            total_style = "bold green" if total_pnl >= 0 else "bold red"
            table.add_row("", "", "", "", "[bold]TOTAL[/]", f"${total_pnl:+,.2f}", "", style=total_style)

        return table

    async def run_live(self, refresh_interval: float = 2.0):
        """
        Run live updating positions display.

        Args:
            refresh_interval: Seconds between updates
        """
        import asyncio

        if not RICH_AVAILABLE or self.console is None:
            print("Live display requires Rich library")
            return

        self.running = True
        self.console.print("[dim]Presione Ctrl+C para salir[/dim]")

        try:
            with Live(self._generate_table(), console=self.console, refresh_per_second=1) as live:
                while self.running:
                    await asyncio.sleep(refresh_interval)
                    live.update(self._generate_table())
        except KeyboardInterrupt:
            self.running = False
            self.console.print("\n[yellow]Detenido[/yellow]")

    def stop(self):
        """Stop live display"""
        self.running = False
