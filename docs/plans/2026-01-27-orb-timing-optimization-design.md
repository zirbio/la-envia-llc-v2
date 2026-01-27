# ORB Timing & Context Optimization Design

**Fecha:** 2026-01-27
**Estado:** Aprobado

## Problema

El flujo actual del bot tiene varios problemas de timing que causan pérdida de oportunidades:

1. **ORB de 15 minutos demasiado lento** - Los mejores breakouts ocurren temprano
2. **Gap de 16 minutos** entre apertura (9:30) y primer monitoreo (9:46)
3. **Polling de 10 segundos** puede perder movimientos rápidos
4. **Inicio no adaptativo** - Si inicias tarde, no se ajusta correctamente
5. **No usa datos de premarket** - Ignora niveles clave de soporte/resistencia
6. **No analiza gap behavior** - No detecta si el gap se llena o expande

## Solución

### 1. Reducir ORB a 5 minutos

| Parámetro | Antes | Después |
|-----------|-------|---------|
| `orb_period_minutes` | 15 | 5 |
| Cálculo ORB | 9:45 | 9:35 |
| Inicio monitoreo | 9:46 | 9:36 |

**Beneficio:** Capturar breakouts entre 9:36-10:00 (ventana más activa).

### 2. Intervalos de monitoreo más rápidos

| Parámetro | Antes | Después |
|-----------|-------|---------|
| `base_monitoring_interval` | 10 seg | 5 seg |
| `min_monitoring_interval` | 5 seg | 2 seg |
| `max_monitoring_interval` | 30 seg | 10 seg |

**Impacto API:** 10 símbolos × 12 checks/min = 120 req/min (límite: 200)

### 3. Lógica de inicio adaptativa

```
< 9:25      → Esperar scheduler normal
9:25-9:30   → Scan ahora, esperar 9:35 para ORB
9:30-9:35   → Scan ahora, esperar 9:35 para ORB
> 9:35      → Scan + ORB + Monitoreo inmediato
> 16:00     → Mensaje "mercado cerrado"
```

## Archivos a Modificar

1. `config/settings.py` - Parámetros de tiempo
2. `main.py` - CronTriggers y `_check_immediate_start()`
3. `data/market_data.py` - Función `get_premarket_data()`
4. `scanner/premarket.py` - Dataclass `PremktContext`
5. `strategy/orb.py` - Context Score calculation

## Timeline Comparativo

```
ANTES:                          DESPUÉS:
09:25 Scan                      09:25 Scan
09:30 Apertura                  09:30 Apertura
09:45 ORB (15 min)              09:35 ORB (5 min)
09:46 Monitoreo                 09:36 Monitoreo
      ↓                               ↓
   16 min perdidos                 6 min perdidos
```

## Context Score (Mejora de Señales)

### 4. Niveles Premarket (0-8 puntos)

Usar High/Low del premarket como filtro de resistencia/soporte.

**Para LONG:**
| Situación | Puntos | Razón |
|-----------|--------|-------|
| ORB High < Premarket High (espacio libre) | +8 | Sin resistencia cercana |
| ORB High ≈ Premarket High (±0.3%) | +4 | Resistencia cerca |
| ORB High > Premarket High | +2 | Ya rompió premarket |

**Para SHORT:**
| Situación | Puntos | Razón |
|-----------|--------|-------|
| ORB Low > Premarket Low (espacio libre) | +8 | Sin soporte cercano |
| ORB Low ≈ Premarket Low (±0.3%) | +4 | Soporte cerca |
| ORB Low < Premarket Low | +2 | Ya rompió premarket |

### 5. Gap Behavior (0-6 puntos)

Detectar si el gap se llena (malo) o se expande (bueno).

**Para LONG (gap up):**
| Situación | Puntos | Razón |
|-----------|--------|-------|
| Precio > Open (gap expande) | +6 | Momentum confirma |
| Precio ≈ Open (±0.5%) | +3 | Neutral |
| Precio < Open (gap llena) | +0 | Fading |

**Para SHORT (gap down):**
| Situación | Puntos | Razón |
|-----------|--------|-------|
| Precio < Open (gap expande) | +6 | Momentum confirma |
| Precio ≈ Open (±0.5%) | +3 | Neutral |
| Precio > Open (gap llena) | +0 | Fading |

### Nuevo Sistema de Puntuación

```
SCORING ACTUAL (0-100):
├─ Breakout strength    0-25 pts
├─ VWAP alignment       0-15 pts
├─ Volume               0-20 pts
├─ RSI                  0-15 pts
├─ MACD                 0-15 pts
└─ Sentiment            0-10 pts

CONTEXT SCORE (+14 pts):
├─ Premarket levels     0-8 pts
└─ Gap behavior         0-6 pts

TOTAL MÁXIMO: 114 pts
```

### Nueva Clasificación de Señales

| Puntuación | Calidad |
|------------|---------|
| ≥ 80 | ÓPTIMA |
| 65-80 | BUENA |
| 50-65 | REGULAR |
| < 50 | DÉBIL |
