"""
fifa_prior.py
=============
Ancla externa de fuerza: puntos del Ranking FIFA (1 abril 2026, última
actualización antes del Mundial). Se usa como PRIOR para que el modelo no se
deje engañar por la fuerza del calendario: equipos que golean en
confederaciones débiles (OFC, AFC menor, CONCACAF menor) aparecen poco en la
base como rivales y, sin este ancla, el modelo los sobrevalora.

Los puntos están mapeados a los nombres CANÓNICOS de tu base de datos.
Cuando reconstruyas la base, basta con actualizar este diccionario (idealmente
con la actualización del 9 de junio de 2026, la última antes del torneo).

Fuente: Ranking Mundial FIFA, abril 2026. Los valores de los equipos por
debajo del puesto ~50 son aproximados (interpolados), suficientes como prior.

NOTA (robustez de nombres): el accesor `fifa_points` ya no exige que el nombre
coincida carácter a carácter con la clave. Normaliza (sin acentos, minúsculas)
y resuelve variantes conocidas (p. ej. 'República Checa' == 'Rep. Checa',
'Bosnia y Herzegovina' == 'Bosnia y Herz.'). Así, da igual la grafía que use
tu base, tus grupos o este diccionario: el prior siempre se encuentra.
"""

from __future__ import annotations

import unicodedata

# Puntos FIFA (abril 2026) por nombre canónico de la base.
FIFA_POINTS: dict[str, float] = {
    "Francia": 1877.0,
    "España": 1876.0,
    "Argentina": 1875.0,
    "Inglaterra": 1826.0,
    "Portugal": 1764.0,
    "Brasil": 1761.0,
    "Países Bajos": 1758.0,
    "Marruecos": 1756.0,
    "Bélgica": 1735.0,
    "Alemania": 1730.0,
    "Croacia": 1717.0,
    "Colombia": 1693.0,
    "Senegal": 1689.0,
    "México": 1681.0,
    "Estados Unidos": 1673.0,
    "Uruguay": 1673.0,
    "Japón": 1660.0,
    "Suiza": 1649.0,
    "Irán": 1615.0,
    "Turquía": 1599.0,
    "Ecuador": 1594.0,
    "Austria": 1593.0,
    "Corea del Sur": 1588.0,
    "Australia": 1580.0,
    "Argelia": 1564.0,
    "Egipto": 1563.0,
    "Canadá": 1556.0,
    "Noruega": 1550.0,
    "Panamá": 1540.0,
    "Costa de Marfil": 1532.0,
    "Suecia": 1514.0,
    "Paraguay": 1503.0,
    "Rep. Checa": 1501.0,
    "Escocia": 1498.0,
    "Túnez": 1483.0,
    "RD Congo": 1478.0,
    "Uzbekistán": 1465.0,
    # Aproximados (puesto > 50):
    "Catar": 1452.0,
    "Irak": 1447.0,
    "Sudáfrica": 1440.0,
    "Arabia Saudita": 1438.0,
    "Jordania": 1433.0,
    "Bosnia y Herz.": 1431.0,
    "Cabo Verde": 1421.0,
    "Ghana": 1409.0,
    "Haití": 1388.0,
    "Curazao": 1386.0,
    "Nueva Zelanda": 1381.0,
}

# Variantes de grafía equivalentes (clave normalizada -> clave en FIFA_POINTS).
# Solo hace falta para nombres que difieren de forma no trivial entre fuentes.
_SPELLING_BRIDGE = {
    "republica checa": "Rep. Checa",
    "czechia": "Rep. Checa",
    "chequia": "Rep. Checa",
    "bosnia y herzegovina": "Bosnia y Herz.",
    "bosnia": "Bosnia y Herz.",
}


def _norm(text: str) -> str:
    if not isinstance(text, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).strip().lower()


# Índice normalizado construido una sola vez.
_NORM_INDEX = {_norm(k): k for k in FIFA_POINTS}


def fifa_points(team_canonical: str) -> float | None:
    """Puntos FIFA del equipo (robusto a acentos y variantes), o None."""
    key = _norm(team_canonical)
    if key in _NORM_INDEX:
        return FIFA_POINTS[_NORM_INDEX[key]]
    if key in _SPELLING_BRIDGE:
        return FIFA_POINTS.get(_SPELLING_BRIDGE[key])
    return None
