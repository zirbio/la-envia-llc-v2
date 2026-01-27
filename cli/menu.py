"""
Menu System for CLI

Provides reusable menu components and navigation utilities.
"""
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from prompt_toolkit import prompt
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False


@dataclass
class MenuItem:
    """Represents a single menu item"""
    key: str
    label: str
    description: str = ""
    action: Optional[Callable] = None
    style: str = "white"


@dataclass
class Menu:
    """Menu configuration and display"""
    title: str
    items: List[MenuItem] = field(default_factory=list)
    border_style: str = "blue"
    show_header: bool = True

    def add_item(
        self,
        key: str,
        label: str,
        description: str = "",
        action: Callable = None,
        style: str = "white"
    ):
        """Add a menu item"""
        self.items.append(MenuItem(
            key=key,
            label=label,
            description=description,
            action=action,
            style=style
        ))

    def add_separator(self):
        """Add a visual separator"""
        self.items.append(MenuItem(key="", label="", description=""))

    def display(self, console: Console = None) -> str:
        """
        Display the menu and get user selection.

        Args:
            console: Rich console (optional)

        Returns:
            Selected key
        """
        if not RICH_AVAILABLE or console is None:
            return self._display_basic()

        return self._display_rich(console)

    def _display_basic(self) -> str:
        """Display menu without Rich"""
        print(f"\n{self.title}")
        print("-" * len(self.title))

        for item in self.items:
            if not item.key:  # Separator
                print()
            else:
                desc = f" - {item.description}" if item.description else ""
                print(f"{item.key}. {item.label}{desc}")

        print()
        return input("Seleccione: ").strip()

    def _display_rich(self, console: Console) -> str:
        """Display menu with Rich"""
        table = Table(
            show_header=self.show_header,
            box=box.ROUNDED,
            border_style=self.border_style,
            padding=(0, 2)
        )

        if self.show_header:
            table.add_column("", style="cyan", width=4)
            table.add_column("Opción", style="white")
            table.add_column("Descripción", style="dim")
        else:
            table.add_column("", style="cyan", width=4)
            table.add_column("")
            table.add_column("", style="dim")

        for item in self.items:
            if not item.key:  # Separator
                table.add_row("", "", "")
            else:
                table.add_row(
                    item.key,
                    item.label,
                    item.description,
                    style=item.style
                )

        panel = Panel(
            table,
            title=f"[bold]{self.title}[/bold]",
            border_style=self.border_style,
            padding=(1, 1)
        )
        console.print(panel)
        console.print()

        if PROMPT_TOOLKIT_AVAILABLE:
            return prompt("Seleccione: ").strip()
        return input("Seleccione: ").strip()

    def get_valid_keys(self) -> List[str]:
        """Get list of valid selection keys"""
        return [item.key for item in self.items if item.key]


class MenuNavigator:
    """Handles menu navigation and history"""

    def __init__(self, console: Console = None):
        self.console = console
        self.history: List[str] = []

    def push(self, menu_name: str):
        """Push menu to history stack"""
        self.history.append(menu_name)

    def pop(self) -> Optional[str]:
        """Pop and return last menu from history"""
        if self.history:
            return self.history.pop()
        return None

    def current(self) -> Optional[str]:
        """Get current menu name"""
        if self.history:
            return self.history[-1]
        return None

    def clear(self):
        """Clear navigation history"""
        self.history.clear()


def create_trading_mode_menu() -> Menu:
    """Create the trading mode selection menu"""
    menu = Menu(
        title="SELECCIONE MODO DE TRADING",
        border_style="blue"
    )

    menu.add_item("1", "Regular Hours", "9:30 - 16:00 EST")
    menu.add_item("2", "Premarket", "8:00 - 9:25 EST")
    menu.add_item("3", "Postmarket", "16:05 - 18:00 EST")
    menu.add_item("4", "Todas las Sesiones", "8:00 - 18:00 EST")
    menu.add_separator()
    menu.add_item("5", "Configuración avanzada")
    menu.add_item("6", "Estado de cuenta")
    menu.add_item("7", "Salir")

    return menu


def create_config_menu(current_level: str) -> Menu:
    """Create the configuration menu"""
    menu = Menu(
        title=f"CONFIGURACIÓN (Nivel: {current_level})",
        border_style="yellow"
    )

    menu.add_item("1", "Cambiar nivel de señal")
    menu.add_item("2", "Ver parámetros actuales")
    menu.add_item("3", "Volver")

    return menu


def create_signal_level_menu() -> Menu:
    """Create the signal level selection menu"""
    menu = Menu(
        title="NIVEL DE SEÑAL",
        border_style="green",
        show_header=True
    )

    menu.add_item("1", "STRICT", "Conservador, alta confianza", style="red")
    menu.add_item("2", "MODERATE", "Balanceado (default)", style="yellow")
    menu.add_item("3", "RELAXED", "Agresivo, más señales", style="green")
    menu.add_separator()
    menu.add_item("4", "Cancelar")

    return menu
