"""
tournament_simulator.py
========================
Montecarlo del MUNDIAL 2026 COMPLETO (formato oficial de 48 equipos):

  - 12 grupos de 4, todos contra todos (3 partidos c/u).
  - Clasifican los 2 primeros de cada grupo + los 8 mejores terceros -> Ronda de 32.
  - Eliminatorias: Ronda de 32 -> 16avos... perdón, R32 -> R16 -> Cuartos -> Semis -> Final.
  - Empates en eliminatoria: prórroga + penales (modelados como desempate
    sesgado por la fuerza relativa de cada equipo).

Detalles de fidelidad:
  - Desempates de grupo: puntos -> diferencia de gol -> goles a favor -> azar.
  - Cuadro de la Ronda de 32: estructura OFICIAL (partidos 73-88), incluyendo
    los grupos elegibles para cada cupo de tercero. Los 8 terceros se asignan a
    sus 8 llaves con un emparejamiento válido que respeta esos grupos.
  - Ventaja de local para las sedes (México, EE.UU., Canadá), opcional.

Genera, para cada equipo: probabilidad de pasar de grupo, llegar a cada ronda,
y ganar el título. Usa el mismo motor (Poisson + Dixon-Coles) ya calibrado.
"""

from __future__ import annotations

import numpy as np

from data_manager import DataManager
from predictive_engine import StatisticalPredictor

HOSTS = {"México", "Estados Unidos", "Canadá"}

# Cuadro oficial de la Ronda de 32 (partidos 73-88).
# Cada entrada: (lado_local, lado_visitante). 'W_X'=ganador grupo X,
# 'R_X'=segundo grupo X, ('3', [grupos]) = mejor tercero de esos grupos.
R32_MATCHES = {
    73: ("R_A", "R_B"),
    74: ("W_E", ("3", list("ABCDF"))),
    75: ("W_F", "R_C"),
    76: ("W_C", "R_F"),
    77: ("W_I", ("3", list("CDFGH"))),
    78: ("R_E", "R_I"),
    79: ("W_A", ("3", list("CEFHI"))),
    80: ("W_L", ("3", list("EHIJK"))),
    81: ("W_D", ("3", list("BEFIJ"))),
    82: ("W_G", ("3", list("AEHIJ"))),
    83: ("R_K", "R_L"),
    84: ("W_H", "R_J"),
    85: ("W_B", ("3", list("EFGIJ"))),
    86: ("W_J", "R_H"),
    87: ("W_K", ("3", list("DEIJL"))),
    88: ("R_D", "R_G"),
}

# Árbol de eliminatorias (de R32 hacia la final). R16 empareja ganadores de
# estos partidos de R32; luego cuartos, semis y final. La R32 es oficial; el
# árbol posterior usa un reparto balanceado en dos mitades (los dos cabezas de
# serie quedan en mitades opuestas, sólo pueden cruzarse en la final).
R16_PAIRS = [
    (73, 75), (74, 77), (76, 78), (79, 80),
    (81, 82), (83, 84), (85, 87), (86, 88),
]
# Índices dentro de R16_PAIRS que se cruzan en cada ronda siguiente.
QF_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]   # ganadores de R16 -> cuartos
SF_PAIRS = [(0, 2), (1, 3)]                    # cuartos -> semis (mitades opuestas)
FINAL_PAIR = (0, 1)                            # semis -> final


def _bipartite_match(slots_eligible: list[set], items_group: list[str]) -> list[int] | None:
    """
    Asigna cada 'item' (tercero, con su grupo) a un 'slot' cuya lista de grupos
    elegibles lo permita, uno a uno (matching perfecto). Devuelve, por slot, el
    índice del item asignado, o None si no hay matching perfecto.
    Augmenting-path simple (suficiente para 8x8).
    """
    n = len(slots_eligible)
    match_slot = [-1] * n      # item asignado a cada slot
    match_item = [-1] * len(items_group)

    def try_assign(slot, seen):
        for it, g in enumerate(items_group):
            if g in slots_eligible[slot] and not seen[it]:
                seen[it] = True
                if match_item[it] == -1 or try_assign(match_item[it], seen):
                    match_slot[slot] = it
                    match_item[it] = slot
                    return True
        return False

    for slot in range(n):
        seen = [False] * len(items_group)
        if not try_assign(slot, seen):
            return None
    return match_slot


class TournamentSimulator:
    def __init__(self, predictor: StatisticalPredictor, groups: dict[str, list[str]],
                 host_advantage: bool = True):
        self.pred = predictor
        self.groups = groups
        self.host_advantage = host_advantage

        # Universo de equipos del torneo y su índice.
        self.teams = [t for g in groups.values() for t in g]
        self.canon = {t: predictor.data.resolve_name(t) for t in self.teams}
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.n_teams = len(self.teams)

        # Precalcula matriz de prob. de victoria en ELIMINATORIA (incl. pen.).
        self._precompute_knockout_winprob()
        # Precalcula distribuciones de marcador para cada cruce de grupo.
        self._precompute_group_matrices()

    # ---------------------------------------------------------- precálculos
    def _venue(self, a: str, b: str) -> str:
        if not self.host_advantage:
            return "neutral"
        if a in HOSTS and b not in HOSTS:
            return "home_a"
        if b in HOSTS and a not in HOSTS:
            return "home_b"
        return "neutral"

    def _precompute_knockout_winprob(self):
        n = self.n_teams
        self.pwin = np.full((n, n), 0.5)
        for a in self.teams:
            for b in self.teams:
                if a == b:
                    continue
                p = self.pred.predict_match(a, b, venue=self._venue(a, b))
                pa, pb, pd = p.prob_win_a, p.prob_win_b, p.prob_draw
                denom = pa + pb
                tilt = pa / denom if denom > 0 else 0.5  # desempate prórroga/penales
                self.pwin[self.idx[a], self.idx[b]] = pa + pd * tilt

    def _precompute_group_matrices(self):
        """Para cada par dentro de un grupo, guarda la matriz DC aplanada."""
        self.group_pairs = {}
        for g, teams in self.groups.items():
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    a, b = teams[i], teams[j]
                    la, lb = self.pred.calculate_lambdas(a, b, venue=self._venue(a, b))
                    m = self.pred.score_matrix(la, lb).ravel()
                    self.group_pairs[(a, b)] = m
        self.n_goals = self.pred.max_goals + 1

    # ------------------------------------------------------- fase de grupos
    def _simulate_groups(self, N, rng):
        """Devuelve, por grupo, arrays [N,4] con el orden de equipos (índices
        0..3 dentro del grupo) y la 'clave' del tercero para comparar terceros."""
        n = self.n_goals
        group_order = {}       # g -> array [N,4] de posiciones (índice local)
        third_key = {}         # g -> array [N] clave del tercero
        third_team = {}        # g -> array [N] equipo (índice global) del tercero

        for g, teams in self.groups.items():
            pts = np.zeros((N, 4), dtype=np.int32)
            gf = np.zeros((N, 4), dtype=np.int32)
            gc = np.zeros((N, 4), dtype=np.int32)
            pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            for (i, j) in pairs:
                a, b = teams[i], teams[j]
                flat = self.group_pairs[(a, b)]
                draws = rng.choice(flat.size, size=N, p=flat)
                ga, gb = np.divmod(draws, n)
                gf[:, i] += ga; gc[:, i] += gb
                gf[:, j] += gb; gc[:, j] += ga
                pts[:, i] += np.where(ga > gb, 3, np.where(ga == gb, 1, 0))
                pts[:, j] += np.where(gb > ga, 3, np.where(ga == gb, 1, 0))

            gd = gf - gc
            # Clave de orden: pts, dif gol, goles a favor, + ruido para desempate.
            noise = rng.random((N, 4)) * 1e-3
            key = pts * 1e6 + gd * 1e3 + gf + noise
            order = np.argsort(-key, axis=1)  # [N,4] mejor->peor (índice local)
            group_order[g] = order

            # Tercero: posición 2 del orden.
            third_local = order[:, 2]
            rows = np.arange(N)
            t_pts = pts[rows, third_local]
            t_gd = gd[rows, third_local]
            t_gf = gf[rows, third_local]
            third_key[g] = t_pts * 1e6 + t_gd * 1e3 + t_gf + rng.random(N) * 1e-3
            third_team[g] = np.array([self.idx[teams[k]] for k in third_local])

        return group_order, third_key, third_team

    # ----------------------------------------------------------- torneo
    def run(self, N: int = 20000, seed: int | None = None) -> dict:
        rng = np.random.default_rng(seed)
        group_order, third_key, third_team = self._simulate_groups(N, rng)
        glist = list(self.groups.keys())

        # ¿Qué grupos aportan tercero clasificado? (8 mejores por clave)
        keys = np.column_stack([third_key[g] for g in glist])    # [N,12]
        best8_cols = np.argsort(-keys, axis=1)[:, :8]            # [N,8] índices de grupo

        # Acumuladores de resultados.
        stage = {t: dict(grupo=0, r32=0, r16=0, qf=0, sf=0, final=0, campeon=0)
                 for t in self.teams}
        teams_by_global = self.teams

        # Mapas para construir la R32.
        def winner_of(order_g, pos, gi, sim):
            # pos: 0=ganador,1=segundo,2=tercero -> índice global del equipo
            local = group_order[gi][sim, pos]
            return self.idx[self.groups[gi][local]]

        pwin = self.pwin

        for s in range(N):
            qualified_groups = [glist[c] for c in best8_cols[s]]
            qset = set(qualified_groups)

            # Marca clasificados de grupo (1°,2° y terceros clasificados).
            for g in glist:
                w = self.groups[g][group_order[g][s, 0]]
                r = self.groups[g][group_order[g][s, 1]]
                stage[w]["grupo"] += 1
                stage[r]["grupo"] += 1
            for g in qualified_groups:
                t_local = group_order[g][s, 2]
                stage[self.groups[g][t_local]]["grupo"] += 1

            # Asigna los 8 terceros a sus 8 llaves según grupos elegibles.
            third_slots = [m for m, (h, a) in R32_MATCHES.items()
                           if isinstance(a, tuple)]
            slots_elig = []
            for m in third_slots:
                _, spec = R32_MATCHES[m]
                slots_elig.append(set(spec[1]) & qset)
            items_group = qualified_groups[:]  # grupo de cada tercero
            assign = _bipartite_match(slots_elig, items_group)
            if assign is None:
                # Respaldo extremadamente raro: asignación por orden.
                assign = list(range(8))
            slot_third_group = {third_slots[k]: items_group[assign[k]]
                                for k in range(8)}

            # Construye los 32 equipos de la R32 (índice global).
            def resolve_side(side):
                if isinstance(side, tuple):  # tercero
                    return None  # se rellena abajo por partido
                kind, g = side.split("_")
                pos = 0 if kind == "W" else 1
                local = group_order[g][s, pos]
                return self.idx[self.groups[g][local]]

            r32_winner = {}
            for m, (hside, aside) in R32_MATCHES.items():
                ta = resolve_side(hside)
                if isinstance(aside, tuple):
                    g3 = slot_third_group[m]
                    tb = int(third_team[g3][s])
                else:
                    tb = resolve_side(aside)
                # Marca participación en R32.
                stage[teams_by_global[ta]]["r32"] += 1
                stage[teams_by_global[tb]]["r32"] += 1
                # Juega.
                win_a = rng.random() < pwin[ta, tb]
                r32_winner[m] = ta if win_a else tb

            # R16
            r16_winner = []
            for (m1, m2) in R16_PAIRS:
                ta, tb = r32_winner[m1], r32_winner[m2]
                stage[teams_by_global[ta]]["r16"] += 1
                stage[teams_by_global[tb]]["r16"] += 1
                r16_winner.append(ta if rng.random() < pwin[ta, tb] else tb)

            # Cuartos
            qf_winner = []
            for (i, j) in QF_PAIRS:
                ta, tb = r16_winner[i], r16_winner[j]
                stage[teams_by_global[ta]]["qf"] += 1
                stage[teams_by_global[tb]]["qf"] += 1
                qf_winner.append(ta if rng.random() < pwin[ta, tb] else tb)

            # Semis
            sf_winner = []
            for (i, j) in SF_PAIRS:
                ta, tb = qf_winner[i], qf_winner[j]
                stage[teams_by_global[ta]]["sf"] += 1
                stage[teams_by_global[tb]]["sf"] += 1
                sf_winner.append(ta if rng.random() < pwin[ta, tb] else tb)

            # Final
            ta, tb = sf_winner[FINAL_PAIR[0]], sf_winner[FINAL_PAIR[1]]
            stage[teams_by_global[ta]]["final"] += 1
            stage[teams_by_global[tb]]["final"] += 1
            champ = ta if rng.random() < pwin[ta, tb] else tb
            stage[teams_by_global[champ]]["campeon"] += 1

        # Normaliza a porcentajes.
        out = {}
        for t in self.teams:
            out[t] = {k: round(v / N * 100, 2) for k, v in stage[t].items()}
        return out


def summarize(results: dict, top: int = 16) -> str:
    rows = sorted(results.items(), key=lambda kv: -kv[1]["campeon"])
    lines = [f"{'Equipo':18s} {'Pasa1ra':>8s} {'R16':>6s} {'Cuartos':>8s} "
             f"{'Semis':>6s} {'Final':>6s} {'CAMPEÓN':>8s}"]
    for t, r in rows[:top]:
        lines.append(f"{t:18s} {r['grupo']:7.1f}% {r['r16']:5.1f}% {r['qf']:7.1f}% "
                     f"{r['sf']:5.1f}% {r['final']:5.1f}% {r['campeon']:7.1f}%")
    return "\n".join(lines)


if __name__ == "__main__":
    from app_groups import GRUPOS  # se define abajo; o importa de app
    dm = DataManager("base_mundial_2026.xlsx")
    pred = StatisticalPredictor(dm, rho=-0.12, goal_scaling=1.076)
    sim = TournamentSimulator(pred, GRUPOS, host_advantage=True)
    res = sim.run(N=20000, seed=42)
    print(summarize(res, top=20))
