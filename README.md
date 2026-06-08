# Motor de Predicción Estocástica · Mundial 2026

Sistema de predicción de partidos y simulación del torneo completo basado en
**Poisson + corrección de Dixon-Coles**, con ratings ajustados por la fuerza
del rival y anclados al ranking FIFA.

## Arquitectura (capas limpias)

| Archivo | Capa | Responsabilidad |
|---|---|---|
| `data_manager.py` | Datos | Carga la base, normaliza nombres y **estima ataque/defensa de cada equipo** vía MLE Poisson ajustado por rival + prior FIFA. |
| `fifa_prior.py` | Datos | Puntos FIFA de los 48 clasificados (ancla anti-sesgo). |
| `predictive_engine.py` | Motor | Lambdas, matriz de marcadores con Dixon-Coles, predicción analítica exacta y Montecarlo. |
| `calibration.py` | Motor | Estima `goal_scaling` y `rho` ajustados a TUS datos. |
| `tournament_simulator.py` | Motor | Montecarlo del torneo completo (grupos + mejores terceros + eliminatorias con cuadro oficial). |
| `api_connector.py` | APIs | Datos en vivo opcionales (lesiones). Degrada con elegancia. |
| `app_groups.py` | Config | Los 12 grupos oficiales. |
| `app.py` | Interfaz | App Streamlit (partido / tabla de fuerzas / simular torneo). |

## El problema que se corrigió: sesgo de fuerza-de-calendario

Cada equipo tiene ~12 partidos, muchos contra rivales débiles de su zona. Un
modelo de ratios crudos premiaba a quien goleaba a rivales flojos (Japón, Nueva
Zelanda salían como potencias). La solución:

1. **Ratings ajustados por rival** (MLE Poisson tipo Dixon-Coles): los goles se
   explican por `exp(mu + ataque_i − defensa_j + ventaja_local)`, así que golear
   a un rival flojo ya no infla tu ataque de forma artificial.
2. **Prior FIFA sobre la FUERZA** (`atk + def`, no `atk − def`): el ranking FIFA
   ancla la potencia real de cada equipo, corrigiendo lo que los datos no pueden
   distinguir ("fuerte" vs "jugó contra débiles"). El estilo (ofensivo/defensivo)
   se encoge para estabilizar.
3. **Calibración**: `goal_scaling` y `rho` se ajustan para reproducir el total de
   goles (~2.59) y la tasa de 0-0 (~5%) observados entre clasificados. Ojo: con
   esta base, los datos NO piden inflar empates (`rho ≈ 0`); forzar un `rho` muy
   negativo dejaba el **1-1 pegado como marcador modal** de casi todos los cruces.

## Uso

```bash
pip install -r requirements.txt
python calibration.py            # parámetros recomendados (goal_scaling, rho)
python tournament_simulator.py   # Montecarlo del torneo completo
streamlit run app.py             # interfaz interactiva
```

## Cuando reconstruyas la base con fechas

- El **decaimiento temporal** se activa solo si añades una columna `Fecha`
  (los partidos recientes pesan más; ver `HALF_LIFE_DAYS` en `data_manager.py`).
- Actualiza `fifa_prior.py` con el ranking FIFA más reciente (idealmente el del
  9 de junio de 2026) para refrescar el ancla.
- Vuelve a correr `calibration.py` y pega los nuevos `goal_scaling`/`rho`.

## Cambios recientes (puesta a punto)

- **Nombres robustos.** La base usa `República Checa` mientras el ranking FIFA
  usaba `Rep. Checa`. Antes, ese desajuste hacía `resolve_name` devolver un
  canónico inexistente → `KeyError` que **tumbaba la tabla de fuerzas, el
  simulador y cualquier predicción de Chequia** (con cualquier peso FIFA).
  Ahora el canónico es SIEMPRE la grafía de tu base y `fifa_prior.fifa_points`
  resuelve variantes/acentos. Da igual cómo escribas cada equipo en cada fuente.
- **Decaimiento temporal (time-decay).** Si la base trae columna `Fecha`, el
  peso de cada partido decae exponencialmente (vida media configurable desde la
  interfaz). Se detecta y se muestra automáticamente.
- **Selección dinámica.** La pestaña de partido ya no se limita a equipos del
  mismo grupo: permite simular **cualquier cruce** (útil para eliminatorias).
- **Marcador modal realista.** La regularización (`RIDGE`) y el encogimiento de
  estilo (`style_shrink`) estaban demasiado altos: comprimían las lambdas y, con
  `rho` negativo, el 1-1 salía en el 98% de los cruces. Con `RIDGE=4`,
  `style_shrink=0.3` y `rho` recalibrado (~0), el modal sigue ahora a la fuerza
  de cada equipo (Francia-Haití 2-1, España-Arabia 1-0, etc.) sin reintroducir
  el sesgo de calendario.
