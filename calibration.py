"""
calibration.py  (v2)
====================
Ajusta los parámetros del motor a TUS datos en vez de fijarlos a ojo:

  - goal_scaling : factor global para que el total de goles esperado del modelo
                   iguale el total observado entre selecciones clasificadas.
  - rho          : corrección de Dixon-Coles para acercar la tasa de 0-0 del
                   modelo a la observada (con un tope para no castigar el Over).

Importante: ahora el ajuste de ratings (MLE) se hace UNA sola vez y se reutiliza
en toda la búsqueda. goal_scaling y rho se aplican después, así que no hay que
re-entrenar nada en cada iteración.

Uso:
    python calibration.py
"""

from __future__ import annotations

import itertools

import numpy as np

from data_manager import DataManager
from predictive_engine import StatisticalPredictor


def observed_targets(dm: DataManager) -> dict:
    """Métricas objetivo medidas en partidos clasificado vs clasificado."""
    df = dm.df.copy()
    qualified = set(dm.list_teams())

    def to_canon(name):
        try:
            return dm.resolve_name(name)
        except ValueError:
            return None

    df["_rival"] = df["Rival (Equipo 2)"].apply(to_canon)
    peer = df[df["_rival"].isin(qualified)]
    if peer.empty:
        peer = df

    gf, gc = peer["Goles F"].to_numpy(), peer["Goles C"].to_numpy()
    total = gf + gc
    return {
        "n_partidos": len(peer),
        "total_goles": float(total.mean()),
        "over_2_5": float((total >= 3).mean()),
        "cero_cero": float(((gf == 0) & (gc == 0)).mean()),
    }


def _model_means(dm: DataManager, gamma: float, rho: float) -> tuple[float, float]:
    """Total de goles medio y tasa de 0-0 media sobre todos los cruces."""
    pred = StatisticalPredictor(dm, rho=rho, goal_scaling=gamma)
    totals, zeros = [], []
    for a, b in itertools.combinations(dm.list_teams(), 2):
        la, lb = pred.calculate_lambdas(a, b)
        m = pred.score_matrix(la, lb)
        totals.append(la + lb)
        zeros.append(float(m[0, 0]))
    return float(np.mean(totals)), float(np.mean(zeros))


def _bisect(fn, target, lo, hi, iters=40):
    """Búsqueda binaria robusta sobre una función monótona (creciente o
    decreciente). Detecta la dirección a partir de los extremos."""
    f_lo, f_hi = fn(lo), fn(hi)
    increasing = f_hi >= f_lo
    for _ in range(iters):
        mid = (lo + hi) / 2
        fm = fn(mid)
        if (fm > target) == increasing:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def calibrate(database_path, rho_floor: float = -0.12) -> dict:
    """Estima goal_scaling y rho ajustados a los datos."""
    dm = database_path if isinstance(database_path, DataManager) else DataManager(database_path)
    tgt = observed_targets(dm)

    gamma = _bisect(
        lambda g: _model_means(dm, g, -0.05)[0], tgt["total_goles"], 0.40, 1.30
    )
    gamma = round(gamma, 3)

    rho_raw = _bisect(
        lambda r: _model_means(dm, gamma, r)[1], tgt["cero_cero"], rho_floor, 0.0
    )
    rho = round(max(rho_raw, rho_floor), 3)

    tot_fit, zero_fit = _model_means(dm, gamma, rho)
    return {
        "targets": tgt,
        "goal_scaling": gamma,
        "rho": rho,
        "ajuste_total_goles": round(tot_fit, 3),
        "ajuste_cero_cero": round(zero_fit, 3),
    }


if __name__ == "__main__":
    dm = DataManager("base_mundial_2026.xlsx")
    res = calibrate(dm)
    t = res["targets"]
    print("=== OBJETIVOS OBSERVADOS (clasificado vs clasificado) ===")
    print(f"  Partidos analizados: {t['n_partidos']}")
    print(f"  Total de goles:      {t['total_goles']:.3f}")
    print(f"  Over 2.5:            {t['over_2_5']*100:.1f}%")
    print(f"  0-0:                 {t['cero_cero']*100:.1f}%")
    print("\n=== PARÁMETROS RECOMENDADOS ===")
    print(f"  goal_scaling = {res['goal_scaling']}")
    print(f"  rho          = {res['rho']}")
    print(f"  (modelo calibrado -> total {res['ajuste_total_goles']}, "
          f"0-0 {res['ajuste_cero_cero']*100:.1f}%)")
