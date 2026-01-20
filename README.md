# Alpaca ORB Trading Bot

Bot de trading semi-autónomo que implementa la estrategia Opening Range Breakout (ORB) para acciones estadounidenses. Escanea gappers pre-market, calcula rangos de apertura, detecta señales de breakout con confirmación multi-indicador y envía alertas vía Telegram para confirmación manual antes de ejecutar.

## Características

- **Escaneo Pre-Market**: Detecta gaps con filtros de volumen, precio y liquidez
- **Estrategia ORB**: Calcula el rango de apertura (primeros 15 minutos) y detecta breakouts
- **Confirmación Multi-Indicador**: VWAP, RSI, MACD, volumen y análisis de sentimiento
- **Alertas Telegram**: Notificaciones en tiempo real con botones de confirmación
- **Gestión de Riesgo**: Kelly Criterion para sizing, stop loss automático, ratio 2:1
- **Paper Trading**: Modo seguro para pruebas con Alpaca Paper

## Requisitos

- Python 3.10+
- Cuenta Alpaca (Paper o Live)
- Bot de Telegram
- API Key de Finnhub (opcional, para sentimiento)

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/la-envia-v2.git
cd la-envia-v2

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales
```

## Configuración

Edita el archivo `.env` con tus credenciales:

```bash
# Alpaca API
ALPACA_API_KEY=tu_api_key
ALPACA_SECRET_KEY=tu_secret_key
ALPACA_PAPER=true

# Telegram
TELEGRAM_BOT_TOKEN=tu_bot_token
TELEGRAM_CHAT_ID=tu_chat_id

# Sentimiento (opcional)
FINNHUB_API_KEY=tu_finnhub_key
SENTIMENT_ENABLED=true
```

### Obtener Credenciales

1. **Alpaca**: Regístrate en [alpaca.markets](https://alpaca.markets) y genera API keys en Paper Trading
2. **Telegram**: Crea un bot con [@BotFather](https://t.me/BotFather) y obtén tu chat ID con [@userinfobot](https://t.me/userinfobot)
3. **Finnhub**: Regístrate en [finnhub.io](https://finnhub.io) para obtener API key gratuita

## Uso

```bash
# Ejecutar el bot
python main.py

# Tests
python test_live.py       # Test completo del scanner
python test_signal.py     # Test de generación de señales
python test_alpaca.py     # Test de conectividad API
python test_telegram.py   # Test del bot de Telegram
```

## Comandos Telegram

| Comando | Descripción |
|---------|-------------|
| `/start` | Iniciar el bot |
| `/stop` | Detener el bot |
| `/status` | Ver estado actual |
| `/watchlist` | Ver candidatos del día |
| `/positions` | Ver posiciones abiertas |
| `/close` | Cerrar todas las posiciones |
| `/help` | Lista de comandos |

## Estrategia ORB

### Señal LONG
- Precio rompe **por encima** del ORB High
- Precio está **por encima** del VWAP
- Volumen actual **≥ 1.5x** promedio
- RSI **< 70** (no sobrecomprado)
- MACD **alcista**
- Sentimiento **≥ -0.3**

### Señal SHORT
- Precio rompe **por debajo** del ORB Low
- Precio está **por debajo** del VWAP
- Volumen actual **≥ 1.5x** promedio
- RSI **> 30** (no sobrevendido)
- MACD **bajista**
- Sentimiento **≤ 0.3**

## Gestión de Riesgo

| Parámetro | Valor |
|-----------|-------|
| Capital | $25,000 |
| Riesgo por trade | 2% |
| Max trades diarios | 3 |
| Stop Loss | Lado opuesto del ORB |
| Take Profit | Ratio 2:1 |
| Position Sizing | Kelly Criterion (half-Kelly) |

## Horario de Operación (EST)

| Hora | Evento |
|------|--------|
| 09:25 | Escaneo pre-market con sentimiento |
| 09:45 | Cálculo del Opening Range |
| 09:46 | Inicio monitoreo de breakouts |
| 16:00 | Cierre de posiciones y resumen diario (market close) |

## Arquitectura

```
la-envia-v2/
├── config/
│   └── settings.py          # Configuración centralizada
├── data/
│   ├── market_data.py       # Cliente Alpaca API
│   ├── indicators.py        # Indicadores técnicos
│   └── sentiment.py         # Análisis de sentimiento
├── scanner/
│   └── premarket.py         # Escáner pre-market
├── strategy/
│   └── orb.py               # Estrategia ORB
├── execution/
│   └── orders.py            # Ejecución de órdenes
├── notifications/
│   └── telegram_bot.py      # Bot de Telegram
├── logs/                    # Logs diarios
├── main.py                  # Punto de entrada
└── requirements.txt         # Dependencias
```

## Dependencias

- `alpaca-py` - SDK oficial de Alpaca
- `pandas` / `numpy` - Análisis de datos
- `python-telegram-bot` - Integración Telegram
- `APScheduler` - Programación de tareas
- `loguru` - Logging avanzado
- `aiohttp` - HTTP asíncrono
- `pytz` - Manejo de zonas horarias

## Logs

Los logs se guardan en `logs/trading_YYYY-MM-DD.log` con rotación diaria y retención de 30 días.

- **stdout**: Nivel INFO (monitoreo en tiempo real)
- **Archivo**: Nivel DEBUG (troubleshooting detallado)

## Advertencias

- Este bot es para **propósitos educativos**
- Siempre usa **Paper Trading** primero
- El trading conlleva **riesgo de pérdida**
- No es consejo financiero

## Licencia

MIT
