"""
backtest.py
===========
Validación HONESTA del motor: ¿qué tan buenas son sus probabilidades cuando
predice partidos que NO ha visto?

Método: backtesting con ventana expansiva (rolling-origin).
  1) Se ordenan los partidos clasificado-vs-clasificado por fecha.
  2) Para cada fecha de corte, se entrena el modelo SOLO con partidos
     anteriores (DataManager(max_date=corte)) y se predicen los partidos del
     bloque siguiente. Nada del futuro se filtra al entrenamiento.
  3) Se acumulan las predicciones fuera-de-muestra y se puntúan con reglas
     de puntuación PROPIAS (proper scoring rules), el estándar académico:

     - RPS  (Ranked Probability Score): para 1X2, sensible al orden
       (confundir victoria con empate penaliza menos que con derrota).
     - Brier multiclase: error cuadrático de las probabilidades 1X2.
     - Log-loss (ignorance): castiga fuerte la sobreconfianza.
     - Brier de Over 2.5 y de "ambos marcan": calidad del modelo DE GOLES.

  4) Todo se compara contra dos baselines:
     - Uniforme (33/33/33): el "no saber nada".
     - Frecuencias históricas (proporción de 1/X/2 en el train): el "saber
       solo lo trivial".
     Si el modelo no le gana a esto, sus probabilidades no aportan señal.

  5) Curva de calibración (reliability): se agrupan las predicciones por
     probabilidad pronosticada y se compara con la frecuencia real. Un modelo
     calibrado dice "70%" y acierta ~70% de esas veces. Es LA medida de
     confianza en las probabilidades de goles.

Uso:
    python backtest.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_manager import DataManager
from predictive_engine import StatisticalPredictor

# Parámetros calibrados del motor (los mismos de app.py).
GOAL_SCALING = 1.079
RHO = 0.0


# ----------------------------------------------------------- reglas de score
def rps_1x2(p: np.ndarray, outcome: int) -> float:
    """Ranked Probability Score para [P(gana A), P(empate), P(gana B)].
    outcome: 0 = gana A, 1 = empate, 2 = gana B. Menor = mejor."""
    o = np.zeros(3)
    o[outcome] = 1.0
    cp, co = np.cumsum(p), np.cumsum(o)
    return float(np.sum((cp[:-1] - co[:-1]) ** 2) / 2.0)


def brier_multi(p: np.ndarray, outcome: int) -> float:
    o = np.zeros(len(p))
    o[outcome] = 1.0
    return float(np.mean((p - o) ** 2))


def log_loss(p_outcome: float) -> float:
    return float(-np.log(max(p_outcome, 1e-12)))


def brier_bin(p: float, hit: bool) -> float:
    return float((p - (1.0 if hit else 0.0)) ** 2)


# ----------------------------------------------------------- datos de test
def peer_matches(dm: DataManager) -> pd.DataFrame:
    """Partidos clasificado-vs-clasificado, deduplicados (cada partido aparece
    dos veces en la base: una por cada equipo analizado)."""
    df = dm.df.copy()

    def canon(n):
        try:
            return dm.resolve_name(n)
        except ValueError:
            return None

    df["_rival"] = df["Rival (Equipo 2)"].apply(canon)
    peer = df[df["_rival"].notna()].copy()
    peer["_fecha"] = pd.to_datetime(peer["Fecha"], errors="coerce")
    peer = peer[peer["_fecha"].notna()]

    peer["_key"] = peer.apply(
        lambda r: (
            tuple(sorted([r["País Analizado"], r["_rival"]])),
            r["_fecha"].strftime("%Y-%m-%d"),
        ),
        axis=1,
    )
    peer = peer.drop_duplicates("_key").sort_values("_fecha").reset_index(drop=True)
    return peer


def _venue_from_cond(cond: str) -> str:
    c = str(cond).strip().lower()
    if c == "local":
        return "home_a"
    if c == "visitante":
        return "home_b"
    return "neutral"


# ----------------------------------------------------------- backtest
def run_backtest(
    database_path: str = "base_mundial_2026.xlsx",
    n_folds: int = 4,
    min_train_matches: int = 150,
    fifa_weight: float = 0.55,
) -> dict:
    """
    Backtest con ventana expansiva. Devuelve métricas agregadas, la tabla de
    predicciones fuera-de-muestra y los bins de calibración.
    """
    dm_full = DataManager(database_path, fifa_weight=fifa_weight)
    peer = peer_matches(dm_full)

    # Cortes: cuantiles de fecha que dejan un train mínimo decente.
    fechas = peer["_fecha"]
    qs = np.linspace(0.45, 0.9, n_folds)  # primer corte ~45% de la historia
    cutoffs = sorted({fechas.quantile(q).normalize() for q in qs})

    rows = []
    for k, cut in enumerate(cutoffs):
        nxt = cutoffs[k + 1] if k + 1 < len(cutoffs) else fechas.max() + pd.Timedelta(days=1)
        test = peer[(peer["_fecha"] > cut) & (peer["_fecha"] <= nxt)]
        if test.empty:
            continue

        # Entrena SOLO con el pasado.
        dm_t = DataManager(
            database_path, fifa_weight=fifa_weight, max_date=str(cut.date())
        )
        if len(dm_t.df) < min_train_matches:
            continue
        pred = StatisticalPredictor(dm_t, rho=RHO, goal_scaling=GOAL_SCALING)

        # Baseline de frecuencias del train (desde la mirada del equipo A).
        res_train = dm_t.df["Resultado"].astype(str).str.strip().str.lower()
        f_win = float((res_train == "ganado").mean())
        f_draw = float((res_train == "empatado").mean())
        f_loss = max(1.0 - f_win - f_draw, 1e-9)
        base_freq = np.array([f_win, f_draw, f_loss])
        base_freq = base_freq / base_freq.sum()

        for _, r in test.iterrows():
            a, b = r["País Analizado"], r["_rival"]
            try:
                p = pred.predict_match(a, b, venue=_venue_from_cond(r["Condición"]))
            except Exception:
                continue  # equipo sin datos suficientes antes del corte

            ga, gc = int(r["Goles F"]), int(r["Goles C"])
            outcome = 0 if ga > gc else (1 if ga == gc else 2)
            total = ga + gc

            p1x2 = np.array([p.prob_win_a, p.prob_draw, p.prob_win_b])
            rows.append({
                "fecha": r["_fecha"].date(),
                "fold": k + 1,
                "equipo_a": a, "equipo_b": b,
                "marcador": f"{ga}-{gc}",
                "outcome": outcome,
                "p_win_a": p.prob_win_a, "p_draw": p.prob_draw, "p_win_b": p.prob_win_b,
                "p_outcome": float(p1x2[outcome]),
                "rps": rps_1x2(p1x2, outcome),
                "brier": brier_multi(p1x2, outcome),
                "logloss": log_loss(float(p1x2[outcome])),
                "rps_unif": rps_1x2(np.array([1/3, 1/3, 1/3]), outcome),
                "brier_unif": brier_multi(np.array([1/3, 1/3, 1/3]), outcome),
                "rps_freq": rps_1x2(base_freq, outcome),
                "brier_freq": brier_multi(base_freq, outcome),
                # Mercados de goles (lo que más te importa):
                "p_over25": p.prob_over_2_5,
                "over25_real": bool(total >= 3),
                "brier_over": brier_bin(p.prob_over_2_5, total >= 3),
                "brier_over_base": brier_bin(0.5, total >= 3),
                "p_btts": p.prob_both_to_score,
                "btts_real": bool(ga > 0 and gc > 0),
                "brier_btts": brier_bin(p.prob_both_to_score, ga > 0 and gc > 0),
                "lam_total": p.lambda_a + p.lambda_b,
                "goles_reales": total,
            })

    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError("No se generaron predicciones fuera-de-muestra.")

    # ---------------- calibración del Over 2.5 (bins de probabilidad)
    bins = np.linspace(0.0, 1.0, 6)  # 5 bins de 20%
    res["_bin"] = pd.cut(res["p_over25"], bins, include_lowest=True)
    calib = (
        res.groupby("_bin", observed=True)
        .agg(n=("over25_real", "size"),
             p_media=("p_over25", "mean"),
             freq_real=("over25_real", "mean"))
        .reset_index()
    )

    # Acierto "duro" del 1X2 (solo informativo; las probabilidades importan más)
    pred_label = res[["p_win_a", "p_draw", "p_win_b"]].to_numpy().argmax(axis=1)
    hit_rate = float((pred_label == res["outcome"].to_numpy()).mean())

    summary = {
        "n_predicciones": len(res),
        "folds": int(res["fold"].nunique()),
        # 1X2
        "rps_modelo": round(float(res["rps"].mean()), 4),
        "rps_uniforme": round(float(res["rps_unif"].mean()), 4),
        "rps_frecuencias": round(float(res["rps_freq"].mean()), 4),
        "brier_modelo": round(float(res["brier"].mean()), 4),
        "brier_uniforme": round(float(res["brier_unif"].mean()), 4),
        "brier_frecuencias": round(float(res["brier_freq"].mean()), 4),
        "logloss_modelo": round(float(res["logloss"].mean()), 4),
        "logloss_uniforme": round(float(np.log(3)), 4),
        "acierto_1x2": round(hit_rate * 100, 1),
        # Goles
        "brier_over25_modelo": round(float(res["brier_over"].mean()), 4),
        "brier_over25_50pct": round(float(res["brier_over_base"].mean()), 4),
        "brier_btts_modelo": round(float(res["brier_btts"].mean()), 4),
        "goles_predichos_media": round(float(res["lam_total"].mean()), 3),
        "goles_reales_media": round(float(res["goles_reales"].mean()), 3),
    }
    return {"summary": summary, "detalle": res, "calibracion_over25": calib}


def print_report(out: dict) -> None:
    s = out["summary"]
    print("=" * 64)
    print("BACKTEST FUERA-DE-MUESTRA (ventana expansiva, sin fugas)")
    print("=" * 64)
    print(f"Predicciones evaluadas: {s['n_predicciones']}  (folds: {s['folds']})")
    print()
    print("--- 1X2 (victoria/empate/derrota) · menor = mejor ---")
    print(f"  RPS    modelo: {s['rps_modelo']:.4f} | uniforme: {s['rps_uniforme']:.4f} "
          f"| frecuencias: {s['rps_frecuencias']:.4f}")
    print(f"  Brier  modelo: {s['brier_modelo']:.4f} | uniforme: {s['brier_uniforme']:.4f} "
          f"| frecuencias: {s['brier_frecuencias']:.4f}")
    print(f"  LogLoss modelo: {s['logloss_modelo']:.4f} | uniforme: {s['logloss_uniforme']:.4f}")
    print(f"  Acierto duro (resultado más probable): {s['acierto_1x2']}%")
    print()
    print("--- Mercados de GOLES · menor = mejor ---")
    print(f"  Brier Over 2.5  modelo: {s['brier_over25_modelo']:.4f} "
          f"| apostar 50% siempre: {s['brier_over25_50pct']:.4f}")
    print(f"  Brier Ambos-marcan modelo: {s['brier_btts_modelo']:.4f}")
    print(f"  Goles por partido -> predichos: {s['goles_predichos_media']} "
          f"| reales: {s['goles_reales_media']}")
    print()
    print("--- Calibración Over 2.5 (p. pronosticada vs frecuencia real) ---")
    for _, r in out["calibracion_over25"].iterrows():
        if r["n"] == 0:
            continue
        print(f"  pronóstico ~{r['p_media']*100:5.1f}%  ->  real {r['freq_real']*100:5.1f}%   "
              f"(n={int(r['n'])})")
    print()
    print("Lectura: si el modelo gana a ambos baselines en RPS/Brier y la")
    print("columna 'real' sigue a la 'pronosticada', tus probabilidades son")
    print("confiables. Si un bin se desvía mucho, ahí hay sobre/sub-confianza.")


if __name__ == "__main__":
    out = run_backtest()
    print_report(out)
