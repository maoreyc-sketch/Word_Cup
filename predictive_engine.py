"""
predictive_engine.py  (v2)
==========================
Capa del Motor Predictivo del Mundial 2026.

Cambios respecto a la v1:
  - Las lambdas (goles esperados) ahora vienen del modelo de ratings AJUSTADO
    POR RIVAL de la capa de datos (DataManager.base_lambdas), que descuenta la
    fuerza del calendario y se ancla en el ranking FIFA. La ventaja de localía
    también se estima de los datos (ya no es una constante a ojo).
  - Se conserva todo lo bueno de la v1: corrección de Dixon-Coles, ajuste por
    arco en cero, predicción analítica exacta y Montecarlo consistente.

Depende de DataManager. No carga archivos por sí mismo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

from data_manager import DataManager

# Rho de Dixon-Coles. Negativo => sube la probabilidad de marcadores bajos y
# empates. calibration.py lo re-estima a partir de tus datos.
DEFAULT_RHO = -0.08

# Cuánto puede reducir la tasa de arcos en cero del rival el lambda propio.
# Se mantiene bajo: los arcos en cero ya están parcialmente reflejados en los
# goles recibidos del rival (evitamos contarlos dos veces).
CLEAN_SHEET_FACTOR = 0.10


@dataclass
class MatchPrediction:
    team_a: str
    team_b: str
    lambda_a: float
    lambda_b: float
    prob_win_a: float
    prob_draw: float
    prob_win_b: float
    prob_over_2_5: float
    prob_both_to_score: float
    most_likely_score: tuple[int, int]
    score_matrix: np.ndarray

    def as_dict(self) -> dict:
        return {
            "prob_local": round(self.prob_win_a * 100, 2),
            "prob_empate": round(self.prob_draw * 100, 2),
            "prob_visitante": round(self.prob_win_b * 100, 2),
            "prob_over_2_5": round(self.prob_over_2_5 * 100, 2),
            "prob_ambos_marcan": round(self.prob_both_to_score * 100, 2),
            "marcador_exacto": self.most_likely_score,
            "lambda_local": round(self.lambda_a, 3),
            "lambda_visitante": round(self.lambda_b, 3),
        }


def _dixon_coles_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


class StatisticalPredictor:
    """Cerebro estadístico: lambdas (ajustadas por rival) y distribuciones."""

    def __init__(
        self,
        database_path,
        rho: float = DEFAULT_RHO,
        max_goals: int = 10,
        goal_scaling: float = 1.0,
        goal_environment: float = 1.0,
    ):
        # Acepta un DataManager ya construido o una ruta de archivo.
        # OJO: en Streamlit Cloud, st.cache_resource conserva objetos entre
        # reruns mientras los módulos se re-importan; eso hace que isinstance
        # falle aunque el objeto SEA un DataManager (la clase ya no es el
        # mismo objeto en memoria). Por eso añadimos duck typing: si camina
        # como DataManager (tiene base_lambdas), lo tratamos como tal.
        if isinstance(database_path, DataManager) or hasattr(database_path, "base_lambdas"):
            self.data = database_path
        else:
            self.data = DataManager(database_path)
        self.rho = rho
        self.max_goals = max_goals
        self.goal_scaling = goal_scaling      # CALIBRACIÓN (ajuste a datos)
        self.goal_environment = goal_environment  # CREENCIA (torneo más abierto)

    # ------------------------------------------------------------ lambdas
    def calculate_lambdas(self, team_a, team_b, venue: str = "neutral"):
        """Expectativa de goles de cada equipo (parámetros del Poisson)."""
        # Base: ratings ajustados por rival + localía estimada de los datos.
        lam_a, lam_b = self.data.base_lambdas(team_a, team_b, venue)

        # Ajuste por arco en cero del rival (defensa sólida => menos goles).
        m_a = self.data.get_team_metrics(team_a)
        m_b = self.data.get_team_metrics(team_b)
        lam_a *= (1.0 - CLEAN_SHEET_FACTOR * m_b.clean_sheet_rate)
        lam_b *= (1.0 - CLEAN_SHEET_FACTOR * m_a.clean_sheet_rate)

        # Calibración (goal_scaling) y creencia (goal_environment).
        factor = self.goal_scaling * self.goal_environment
        lam_a *= factor
        lam_b *= factor

        return max(lam_a, 1e-6), max(lam_b, 1e-6)

    # ------------------------------------------------------- matriz exacta
    def score_matrix(self, lam_a: float, lam_b: float) -> np.ndarray:
        n = self.max_goals + 1
        pa = poisson.pmf(np.arange(n), lam_a)
        pb = poisson.pmf(np.arange(n), lam_b)
        matrix = np.outer(pa, pb)
        for x in (0, 1):
            for y in (0, 1):
                matrix[x, y] *= _dixon_coles_tau(x, y, lam_a, lam_b, self.rho)
        total = matrix.sum()
        return matrix / total if total > 0 else matrix

    # ----------------------------------------------------- predicción exacta
    def predict_match(self, team_a, team_b, venue: str = "neutral") -> MatchPrediction:
        lam_a, lam_b = self.calculate_lambdas(team_a, team_b, venue)
        m = self.score_matrix(lam_a, lam_b)

        prob_a = float(np.tril(m, -1).sum())
        prob_draw = float(np.trace(m))
        prob_b = float(np.triu(m, 1).sum())

        idx = np.arange(self.max_goals + 1)
        gx, gy = np.meshgrid(idx, idx, indexing="ij")
        prob_over = float(m[(gx + gy) >= 3].sum())
        prob_btts = float(m[1:, 1:].sum())

        i, j = np.unravel_index(np.argmax(m), m.shape)

        return MatchPrediction(
            team_a=self.data.resolve_name(team_a),
            team_b=self.data.resolve_name(team_b),
            lambda_a=lam_a,
            lambda_b=lam_b,
            prob_win_a=prob_a,
            prob_draw=prob_draw,
            prob_win_b=prob_b,
            prob_over_2_5=prob_over,
            prob_both_to_score=prob_btts,
            most_likely_score=(int(i), int(j)),
            score_matrix=m,
        )

    # ------------------------------------------------- Montecarlo (1 partido)
    def run_monte_carlo_simulation(
        self, team_a, team_b, simulations: int = 10000,
        venue: str = "neutral", rng=None,
    ) -> dict:
        rng = rng or np.random.default_rng()
        lam_a, lam_b = self.calculate_lambdas(team_a, team_b, venue)
        m = self.score_matrix(lam_a, lam_b)

        n = self.max_goals + 1
        flat = m.ravel()
        draws = rng.choice(flat.size, size=simulations, p=flat)
        goals_a, goals_b = np.divmod(draws, n)

        wins_a = int(np.sum(goals_a > goals_b))
        draws_n = int(np.sum(goals_a == goals_b))
        wins_b = int(np.sum(goals_a < goals_b))
        total = goals_a + goals_b

        pair_codes = goals_a * n + goals_b
        modal = np.bincount(pair_codes).argmax()
        ms_a, ms_b = divmod(int(modal), n)

        return {
            "prob_local": round(wins_a / simulations * 100, 2),
            "prob_empate": round(draws_n / simulations * 100, 2),
            "prob_visitante": round(wins_b / simulations * 100, 2),
            "prob_over_2_5": round(np.mean(total >= 3) * 100, 2),
            "prob_ambos_marcan": round(np.mean((goals_a > 0) & (goals_b > 0)) * 100, 2),
            "marcador_exacto": (ms_a, ms_b),
            "lambda_local": round(lam_a, 3),
            "lambda_visitante": round(lam_b, 3),
        }

    # ------------------------------------------- utilidades para el torneo
    def sample_scores(self, team_a, team_b, size, venue="neutral", rng=None):
        """Muestrea 'size' marcadores (goles_a, goles_b) de la matriz DC.
        Vectorizado: pensado para simular muchos partidos del torneo."""
        rng = rng or np.random.default_rng()
        lam_a, lam_b = self.calculate_lambdas(team_a, team_b, venue)
        m = self.score_matrix(lam_a, lam_b)
        n = self.max_goals + 1
        draws = rng.choice(m.size, size=size, p=m.ravel())
        ga, gb = np.divmod(draws, n)
        return ga, gb


if __name__ == "__main__":
    # Demo con los parámetros calibrados por calibration.py sobre esta base.
    pred = StatisticalPredictor("base_mundial_2026.xlsx", rho=0.0, goal_scaling=1.079)
    print("=== MÉTODO ANALÍTICO (exacto) ===")
    for k, v in pred.predict_match("Colombia", "Portugal").as_dict().items():
        print(f"  {k}: {v}")
    print("\n=== MONTECARLO (10.000 sims) ===")
    for k, v in pred.run_monte_carlo_simulation("Colombia", "Portugal").items():
        print(f"  {k}: {v}")
