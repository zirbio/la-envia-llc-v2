"""
Main Menu Screen

Primary menu for the CLI interface.
"""
from typing import Optional
from datetime import datetime
import pytz

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from prompt_toolkit import prompt
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

from config.settings import TradingMode
from execution.orders import order_executor


EST = pytz.timezone('US/Eastern')
MADRID = pytz.timezone('Europe/Madrid')


def get_current_dual_time() -> str:
    """Get current time in both EST and Madrid timezones."""
    now_est = datetime.now(EST)
    now_madrid = now_est.astimezone(MADRID)
    return f"{now_est.strftime('%H:%M')} EST / {now_madrid.strftime('%H:%M')} Madrid"


class MainMenuScreen:
    """Main menu screen component"""

    def __init__(self, console: Console = None):
        self.console = console

    def get_status_info(self) -> dict:
        """Get current status information"""
        # Market status
        try:
            status = order_executor.get_extended_hours_status()
            if status.get('is_regular'):
                market_status = ("ABIERTO", "green")
            elif status.get('is_premarket'):
                market_status = ("PREMARKET", "yellow")
            elif status.get('is_postmarket'):
                market_status = ("POSTMARKET", "yellow")
            else:
                market_status = ("CERRADO", "red")
        except Exception:
            market_status = ("DESCONOCIDO", "dim")

        # Connection status
        try:
            account = order_executor.get_account()
            conn_status = ("OK", "green") if account else ("ERROR", "red")
        except Exception:
            conn_status = ("ERROR", "red")

        # Account info
        try:
            account = order_executor.get_account()
            equity = account.get('equity', 0) if account else 0
        except Exception:
            equity = 0

        now_est = datetime.now(EST)
        now_madrid = now_est.astimezone(MADRID)

        return {
            'market_status': market_status,
            'connection_status': conn_status,
            'equity': equity,
            'time': f"{now_est.strftime('%H:%M:%S')} EST / {now_madrid.strftime('%H:%M:%S')} Madrid"
        }

    def display(self) -> str:
        """Display main menu and return selection"""
        if not RICH_AVAILABLE or self.console is None:
            return self._display_basic()

        return self._display_rich()

    def _display_basic(self) -> str:
        """Display without Rich"""
        info = self.get_status_info()
        market_text, _ = info['market_status']
        conn_text, _ = info['connection_status']
        current_time = get_current_dual_time()

        print("\n" + "=" * 60)
        print("           ALPACA ORB TRADING BOT")
        print("=" * 60)
        print(f"[Status: Mercado {market_text} | Conexión: {conn_text}]")
        print(f"⏰ Hora actual: {current_time}")
        print()
        print("SELECCIONE MODO DE TRADING:")
        print("1. Regular Hours  9:30-16:00 EST (15:30-22:00 Madrid) <- Default")
        print("2. Premarket      8:00-9:25 EST  (14:00-15:25 Madrid)")
        print("3. Postmarket     16:05-18:00 EST (22:05-00:00 Madrid)")
        print("4. Todas Sesiones 8:00-18:00 EST (14:00-00:00 Madrid)")
        print("-" * 60)
        print("5. Configuración avanzada")
        print("6. Estado de cuenta")
        print("7. Salir")

        return input("\nSeleccione [1-7]: ").strip()

    def _display_rich(self) -> str:
        """Display with Rich"""
        info = self.get_status_info()
        market_text, market_color = info['market_status']
        conn_text, conn_color = info['connection_status']

        # Header
        header = Text()
        header.append("ALPACA ORB TRADING BOT\n", style="bold cyan")
        header.append(f"Mercado: ", style="dim")
        header.append(f"{market_text}", style=market_color)
        header.append(" | ", style="dim")
        header.append("Conexión: ", style="dim")
        header.append(f"{conn_text}", style=conn_color)
        header.append(f" | {info['time']}", style="dim")

        header_panel = Panel(
            header,
            box=box.DOUBLE_EDGE,
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(header_panel)

        # Menu options
        table = Table(
            show_header=False,
            box=box.ROUNDED,
            border_style="blue",
            padding=(0, 2),
            expand=True
        )
        table.add_column("Opción", style="cyan", width=4)
        table.add_column("Descripción", style="white")
        table.add_column("Horario EST", style="dim", justify="right")
        table.add_column("Horario Madrid", style="dim cyan", justify="right")

        # Add Default marker to option 1
        table.add_row("1", "Regular Hours [bold yellow]<- Default[/]", "9:30 - 16:00", "15:30 - 22:00")
        table.add_row("2", "Premarket", "8:00 - 9:25", "14:00 - 15:25")
        table.add_row("3", "Postmarket", "16:05 - 18:00", "22:05 - 00:00")
        table.add_row("4", "Todas las Sesiones", "8:00 - 18:00", "14:00 - 00:00")
        table.add_row("", "", "", "")
        table.add_row("5", "Configuración avanzada", "", "")
        table.add_row("6", "Estado de cuenta", "", "")
        table.add_row("7", "Salir", "", "")

        current_time = get_current_dual_time()
        menu_panel = Panel(
            table,
            title="[bold]SELECCIONE MODO DE TRADING[/bold]",
            subtitle=f"[dim]⏰ Hora actual: {current_time}[/dim]",
            border_style="blue",
            padding=(1, 1)
        )
        self.console.print(menu_panel)
        self.console.print()

        if PROMPT_TOOLKIT_AVAILABLE:
            return prompt("Seleccione [1-7]: ").strip()
        return input("Seleccione [1-7]: ").strip()

    def parse_selection(self, selection: str) -> Optional[TradingMode]:
        """
        Parse menu selection to TradingMode.

        Args:
            selection: User input string

        Returns:
            TradingMode if valid mode selected, None for other options
        """
        mode_map = {
            '1': TradingMode.REGULAR,
            '2': TradingMode.PREMARKET,
            '3': TradingMode.POSTMARKET,
            '4': TradingMode.ALL_SESSIONS
        }
        return mode_map.get(selection)
