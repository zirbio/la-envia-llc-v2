"""
CLI Application Controller

Main controller for the interactive CLI interface.
Uses Rich for beautiful terminal output and prompt_toolkit for input.
"""
import asyncio
import os
from typing import Optional
from datetime import datetime
import pytz

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.live import Live
    from rich.layout import Layout
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from prompt_toolkit import prompt
    from prompt_toolkit.shortcuts import clear
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

from loguru import logger
from config.settings import settings, TradingMode, SignalLevel
from execution.orders import order_executor


EST = pytz.timezone('US/Eastern')
MADRID = pytz.timezone('Europe/Madrid')


def format_dual_timezone(est_time: str) -> str:
    """
    Convert EST time string to display with both EST and Madrid timezones.

    Args:
        est_time: Time in format "HH:MM" (EST)

    Returns:
        String like "9:30 EST (15:30 Madrid)"
    """
    hour, minute = map(int, est_time.replace(' ', '').split(':'))
    now = datetime.now(EST)
    est_dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    madrid_dt = est_dt.astimezone(MADRID)
    return f"{est_time} EST ({madrid_dt.strftime('%H:%M')} Madrid)"


def get_current_dual_time() -> str:
    """Get current time in both EST and Madrid timezones."""
    now_est = datetime.now(EST)
    now_madrid = now_est.astimezone(MADRID)
    return f"{now_est.strftime('%H:%M')} EST / {now_madrid.strftime('%H:%M')} Madrid"


class CLIApp:
    """
    Interactive CLI Application for Trading Bot

    Provides menu-driven interface for:
    - Selecting trading mode
    - Viewing account status
    - Configuring settings
    """

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.selected_mode: Optional[TradingMode] = None
        self.running = True

    def _print(self, text: str, style: str = None):
        """Print with or without Rich"""
        if self.console:
            self.console.print(text, style=style)
        else:
            print(text)

    def _clear(self):
        """Clear screen (cross-platform)"""
        if PROMPT_TOOLKIT_AVAILABLE:
            clear()
        else:
            # Use os.system for cross-platform compatibility
            # 'cls' for Windows, 'clear' for Unix/macOS
            os.system('cls' if os.name == 'nt' else 'clear')

    def _get_market_status(self) -> tuple[str, str]:
        """Get current market status"""
        try:
            status = order_executor.get_extended_hours_status()
            if status.get('is_regular'):
                return "ABIERTO", "green"
            elif status.get('is_premarket'):
                return "PREMARKET", "yellow"
            elif status.get('is_postmarket'):
                return "POSTMARKET", "yellow"
            else:
                return "CERRADO", "red"
        except Exception as e:
            logger.warning(f"Error getting market status: {e}")
            return "DESCONOCIDO", "dim"

    def _get_connection_status(self) -> tuple[str, str]:
        """Check API connection status"""
        try:
            account = order_executor.get_account()
            if account:
                return "OK", "green"
            logger.warning("API connection failed: empty account response")
            return "ERROR", "red"
        except Exception as e:
            logger.warning(f"API connection error: {e}")
            return "ERROR", "red"

    def _display_header(self):
        """Display application header"""
        if not RICH_AVAILABLE:
            print("=" * 50)
            print("       ALPACA ORB TRADING BOT")
            print("=" * 50)
            return

        market_status, market_color = self._get_market_status()
        conn_status, conn_color = self._get_connection_status()

        header = Text()
        header.append("ALPACA ORB TRADING BOT\n", style="bold cyan")
        header.append(f"Mercado: ", style="dim")
        header.append(f"{market_status}", style=market_color)
        header.append(" | ", style="dim")
        header.append("Conexión: ", style="dim")
        header.append(f"{conn_status}", style=conn_color)

        panel = Panel(
            header,
            box=box.DOUBLE_EDGE,
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(panel)

    def _display_main_menu(self) -> str:
        """Display main menu and get selection"""
        current_time = get_current_dual_time()

        if not RICH_AVAILABLE:
            print("\nSELECCIONE MODO DE TRADING:")
            print("1. Regular Hours  9:30-16:00 EST (15:30-22:00 Madrid)")
            print("2. Premarket      8:00-9:25 EST  (14:00-15:25 Madrid)")
            print("3. Postmarket     16:05-18:00 EST (22:05-00:00 Madrid)")
            print("4. Todas Sesiones 8:00-18:00 EST (14:00-00:00 Madrid)")
            print("-" * 50)
            print("5. Configuración avanzada")
            print("6. Estado de cuenta")
            print("7. Salir")
            print(f"\n⏰ Hora actual: {current_time}")
            return input("\nSeleccione [1-7]: ").strip()

        # Create menu table
        table = Table(
            show_header=False,
            box=box.ROUNDED,
            border_style="blue",
            padding=(0, 2),
            expand=True
        )
        table.add_column("Opción", style="cyan", width=4)
        table.add_column("Descripción", style="white")
        table.add_column("Horario EST", style="dim")
        table.add_column("Horario Madrid", style="dim cyan")

        table.add_row("1", "Regular Hours", "9:30 - 16:00", "15:30 - 22:00")
        table.add_row("2", "Premarket", "8:00 - 9:25", "14:00 - 15:25")
        table.add_row("3", "Postmarket", "16:05 - 18:00", "22:05 - 00:00")
        table.add_row("4", "Todas las Sesiones", "8:00 - 18:00", "14:00 - 00:00")
        table.add_row("", "", "", "")
        table.add_row("5", "Configuración avanzada", "", "")
        table.add_row("6", "Estado de cuenta", "", "")
        table.add_row("7", "Salir", "", "")

        panel = Panel(
            table,
            title="[bold]SELECCIONE MODO DE TRADING[/bold]",
            subtitle=f"[dim]⏰ Hora actual: {current_time}[/dim]",
            border_style="blue",
            padding=(1, 1)
        )
        self.console.print(panel)

        # Get input
        self.console.print()
        if PROMPT_TOOLKIT_AVAILABLE:
            return prompt("Seleccione [1-7]: ").strip()
        return input("Seleccione [1-7]: ").strip()

    def _display_config_menu(self) -> str:
        """Display configuration menu"""
        current_level = settings.trading.signal_level.value
        current_exec_mode = settings.trading.execution_mode.upper()

        if not RICH_AVAILABLE:
            print(f"\nCONFIGURACIÓN (Nivel: {current_level} | Ejecución: {current_exec_mode})")
            print("1. Cambiar nivel de señal")
            print("2. Cambiar modo de ejecución (AUTO/MANUAL)")
            print("3. Ver parámetros actuales")
            print("4. Volver")
            return input("\nSeleccione [1-4]: ").strip()

        table = Table(show_header=False, box=box.ROUNDED, border_style="yellow")
        table.add_column("Opción", style="cyan", width=4)
        table.add_column("Descripción")

        exec_mode_desc = f"Cambiar modo de ejecución (actual: {current_exec_mode})"
        table.add_row("1", "Cambiar nivel de señal")
        table.add_row("2", exec_mode_desc)
        table.add_row("3", "Ver parámetros actuales")
        table.add_row("4", "Volver")

        panel = Panel(
            table,
            title=f"[bold]CONFIGURACIÓN[/bold] (Nivel: {current_level} | Ejecución: {current_exec_mode})",
            border_style="yellow"
        )
        self.console.print(panel)

        self.console.print()
        if PROMPT_TOOLKIT_AVAILABLE:
            return prompt("Seleccione [1-4]: ").strip()
        return input("Seleccione [1-4]: ").strip()

    def _display_signal_level_menu(self) -> str:
        """Display signal level selection"""
        if not RICH_AVAILABLE:
            print("\nSELECCIONE NIVEL DE SEÑAL:")
            print("1. STRICT - Conservador, alta confianza")
            print("2. MODERATE - Balanceado (default)")
            print("3. RELAXED - Agresivo, más señales")
            print("4. Cancelar")
            return input("\nSeleccione [1-4]: ").strip()

        table = Table(show_header=True, box=box.ROUNDED, border_style="green")
        table.add_column("", style="cyan", width=4)
        table.add_column("Nivel", style="bold")
        table.add_column("Score Min", justify="center")
        table.add_column("RVOL Min", justify="center")
        table.add_column("Horario Max")

        table.add_row("1", "STRICT", "70", "1.5x", "11:30")
        table.add_row("2", "MODERATE", "55", "1.2x", "14:30")
        table.add_row("3", "RELAXED", "40", "1.0x", "15:30")
        table.add_row("4", "Cancelar", "", "", "")

        panel = Panel(table, title="[bold]NIVEL DE SEÑAL[/bold]", border_style="green")
        self.console.print(panel)

        self.console.print()
        if PROMPT_TOOLKIT_AVAILABLE:
            return prompt("Seleccione [1-4]: ").strip()
        return input("Seleccione [1-4]: ").strip()

    def _display_execution_mode_menu(self) -> str:
        """Display execution mode selection"""
        current_mode = settings.trading.execution_mode.upper()

        if not RICH_AVAILABLE:
            print(f"\nMODO DE EJECUCIÓN (actual: {current_mode})")
            print("1. AUTO - Ejecuta señales automáticamente sin confirmación")
            print("2. MANUAL - Envía alerta a Telegram y espera confirmación SI/NO")
            print("3. Cancelar")
            return input("\nSeleccione [1-3]: ").strip()

        table = Table(show_header=True, box=box.ROUNDED, border_style="magenta")
        table.add_column("", style="cyan", width=4)
        table.add_column("Modo", style="bold")
        table.add_column("Descripción")

        auto_marker = "[bold green]← actual[/]" if current_mode == "AUTO" else ""
        manual_marker = "[bold green]← actual[/]" if current_mode == "MANUAL" else ""

        table.add_row("1", f"AUTO {auto_marker}", "Ejecuta señales automáticamente sin confirmación")
        table.add_row("2", f"MANUAL {manual_marker}", "Envía alerta a Telegram y espera SI/NO")
        table.add_row("3", "Cancelar", "")

        panel = Panel(table, title=f"[bold]MODO DE EJECUCIÓN[/bold]", border_style="magenta")
        self.console.print(panel)

        self.console.print()
        if PROMPT_TOOLKIT_AVAILABLE:
            return prompt("Seleccione [1-3]: ").strip()
        return input("Seleccione [1-3]: ").strip()

    def _display_account_status(self):
        """Display account status"""
        account = order_executor.get_account()
        positions = order_executor.get_positions()

        if not RICH_AVAILABLE:
            print("\n" + "=" * 40)
            print("ESTADO DE CUENTA")
            print("=" * 40)
            if account:
                print(f"Equity: ${account.get('equity', 0):,.2f}")
                print(f"Cash: ${account.get('cash', 0):,.2f}")
                print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
                print(f"Day Trades: {account.get('day_trade_count', 0)}")
            print(f"\nPosiciones: {len(positions)}")
            for pos in positions:
                print(f"  {pos.symbol}: {pos.qty} @ ${pos.entry_price:.2f} ({pos.unrealized_pl_pct:+.1f}%)")
            input("\nPresione Enter para continuar...")
            return

        # Account info table
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

        self.console.print(Panel(account_table, title="[bold]CUENTA[/bold]", border_style="cyan"))

        # Positions table
        if positions:
            pos_table = Table(box=box.ROUNDED, border_style="blue")
            pos_table.add_column("Symbol", style="bold")
            pos_table.add_column("Qty", justify="right")
            pos_table.add_column("Entry", justify="right")
            pos_table.add_column("Current", justify="right")
            pos_table.add_column("P/L", justify="right")
            pos_table.add_column("P/L %", justify="right")

            for pos in positions:
                pnl_style = "green" if pos.unrealized_pl >= 0 else "red"
                pos_table.add_row(
                    pos.symbol,
                    str(pos.qty),
                    f"${pos.entry_price:.2f}",
                    f"${pos.current_price:.2f}",
                    f"${pos.unrealized_pl:+.2f}",
                    f"{pos.unrealized_pl_pct:+.1f}%",
                    style=pnl_style
                )

            self.console.print(Panel(pos_table, title=f"[bold]POSICIONES ({len(positions)})[/bold]", border_style="blue"))
        else:
            self.console.print(Panel("Sin posiciones abiertas", style="dim"))

        self.console.print()
        if PROMPT_TOOLKIT_AVAILABLE:
            prompt("Presione Enter para continuar...")
        else:
            input("Presione Enter para continuar...")

    def _display_current_params(self):
        """Display current trading parameters"""
        config = settings.trading.signal_config
        exec_mode = settings.trading.execution_mode.upper()

        if not RICH_AVAILABLE:
            print(f"\nPARÁMETROS ACTUALES (Nivel: {settings.trading.signal_level.value} | Ejecución: {exec_mode})")
            print(f"  Modo Ejecución: {exec_mode}")
            print(f"  Min Signal Score: {config.min_signal_score}")
            print(f"  Min RVOL: {config.min_relative_volume}x")
            print(f"  ORB Range: {config.min_orb_range_pct}% - {config.max_orb_range_pct}%")
            print(f"  Latest Trade: {config.latest_trade_time}")
            print(f"  RSI Bounds: {config.rsi_oversold} - {config.rsi_overbought}")
            print(f"  Sentiment Long: >= {config.min_sentiment_long}")
            print(f"  Sentiment Short: <= {config.max_sentiment_short}")
            input("\nPresione Enter para continuar...")
            return

        table = Table(show_header=True, box=box.ROUNDED, border_style="magenta")
        table.add_column("Parámetro")
        table.add_column("Valor", justify="right")

        # Execution mode at the top for visibility
        exec_style = "[bold green]AUTO[/]" if exec_mode == "AUTO" else "[bold yellow]MANUAL[/]"
        table.add_row("Modo Ejecución", exec_style)
        table.add_row("", "")  # Separator
        table.add_row("Min Signal Score", str(config.min_signal_score))
        table.add_row("Min RVOL", f"{config.min_relative_volume}x")
        table.add_row("ORB Range Min", f"{config.min_orb_range_pct}%")
        table.add_row("ORB Range Max", f"{config.max_orb_range_pct}%")
        table.add_row("Latest Trade Time", config.latest_trade_time)
        table.add_row("RSI Overbought", str(config.rsi_overbought))
        table.add_row("RSI Oversold", str(config.rsi_oversold))
        table.add_row("Sentiment Long >=", str(config.min_sentiment_long))
        table.add_row("Sentiment Short <=", str(config.max_sentiment_short))
        table.add_row("Require Candle Close", str(config.require_candle_close))

        panel = Panel(
            table,
            title=f"[bold]PARÁMETROS ({settings.trading.signal_level.value})[/bold]",
            border_style="magenta"
        )
        self.console.print(panel)

        self.console.print()
        if PROMPT_TOOLKIT_AVAILABLE:
            prompt("Presione Enter para continuar...")
        else:
            input("Presione Enter para continuar...")

    def _handle_config_menu(self):
        """Handle configuration menu"""
        while True:
            self._clear()
            self._display_header()
            choice = self._display_config_menu()

            if choice == '1':
                self._clear()
                self._display_header()
                level_choice = self._display_signal_level_menu()

                level_map = {
                    '1': SignalLevel.STRICT,
                    '2': SignalLevel.MODERATE,
                    '3': SignalLevel.RELAXED
                }

                if level_choice in level_map:
                    from strategy.orb import orb_strategy
                    orb_strategy.set_signal_level(level_map[level_choice])
                    self._print(f"\n[green]Nivel cambiado a {level_map[level_choice].value}[/green]" if RICH_AVAILABLE
                               else f"\nNivel cambiado a {level_map[level_choice].value}")
                    if PROMPT_TOOLKIT_AVAILABLE:
                        prompt("Presione Enter para continuar...")
                    else:
                        input("Presione Enter para continuar...")

            elif choice == '2':
                self._clear()
                self._display_header()
                exec_choice = self._display_execution_mode_menu()

                exec_map = {
                    '1': 'auto',
                    '2': 'manual'
                }

                if exec_choice in exec_map:
                    settings.trading.execution_mode = exec_map[exec_choice]
                    self._print(f"\n[green]Modo de ejecución cambiado a {exec_map[exec_choice].upper()}[/green]" if RICH_AVAILABLE
                               else f"\nModo de ejecución cambiado a {exec_map[exec_choice].upper()}")
                    if PROMPT_TOOLKIT_AVAILABLE:
                        prompt("Presione Enter para continuar...")
                    else:
                        input("Presione Enter para continuar...")

            elif choice == '3':
                self._clear()
                self._display_header()
                self._display_current_params()

            elif choice == '4' or choice == '':
                break

    async def run(self) -> Optional[TradingMode]:
        """
        Run the interactive CLI.

        Returns:
            Selected TradingMode or None if user exits
        """
        if not RICH_AVAILABLE:
            logger.warning("Rich library not available. Using basic CLI.")
        if not PROMPT_TOOLKIT_AVAILABLE:
            logger.warning("prompt_toolkit not available. Using basic input.")

        while self.running:
            self._clear()
            self._display_header()
            choice = self._display_main_menu()

            mode_map = {
                '1': TradingMode.REGULAR,
                '2': TradingMode.PREMARKET,
                '3': TradingMode.POSTMARKET,
                '4': TradingMode.ALL_SESSIONS
            }

            if choice in mode_map:
                self.selected_mode = mode_map[choice]
                self._print(f"\nIniciando modo: {self.selected_mode.value.upper()}" if RICH_AVAILABLE
                           else f"\nIniciando modo: {self.selected_mode.value.upper()}")
                self.running = False
                return self.selected_mode

            elif choice == '5':
                self._handle_config_menu()

            elif choice == '6':
                self._clear()
                self._display_header()
                self._display_account_status()

            elif choice == '7':
                self._print("\nSaliendo..." if RICH_AVAILABLE else "\nSaliendo...")
                self.running = False
                return None

            else:
                self._print("\n[red]Opción inválida[/red]" if RICH_AVAILABLE else "\nOpción inválida")
                await asyncio.sleep(1)

        return self.selected_mode
