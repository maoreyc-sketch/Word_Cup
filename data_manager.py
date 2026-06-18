"""
data_manager.py  (v2 - robusto)
===============================
Capa de Datos del Motor de Predicción del Mundial 2026.

Cambio de fondo respecto a la v1
--------------------------------
La v1 calculaba la fuerza de ataque/defensa como:
    ataque  = goles_a_favor_medios  / media_global_de_goles_a_favor
    defensa = goles_en_contra_medios / media_global_de_goles_en_contra
Eso NO ajusta por la calidad del rival. Resultado: Nueva Zelanda (que golea en
la OFC) salía con el mejor ataque del mundo, y Ecuador o Paraguay (que solo
juegan rivales durísimos de CONMEBOL) salían débiles. Es el clásico problema
de "strength of schedule".

La v2 estima las fuerzas con un MODELO POISSON AJUSTADO POR RIVAL
(estilo Maher / Dixon-Coles, vía máxima verosimilitud regularizada):

    goles que el equipo i le mete a j  ~  Poisson( exp(mu + atk_i - def_j + h*local_i) )

Se estiman a la vez:
  - mu  : nivel base de goles del dataset,
  - h   : ventaja de localía (estimada de los datos, no fijada a ojo),
  - atk_i, def_i para CADA equipo que aparece (los 48 clasificados Y sus
    rivales no clasificados), de modo que cada rating queda en una escala
    comparable y descontando la calidad de los rivales enfrentados.

Regularización L2 (ridge) sobre atk/def: actúa como un prior que empuja a los
equipos con pocos partidos hacia la media. Esto da robustez y resuelve la
indeterminación del modelo sin tener que imponer restricciones de suma.

Soporta (opcional) decaimiento temporal: si la base trae una columna de fecha,
los partidos viejos pesan menos. Si no hay fecha, el sistema funciona igual.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ponderaciones por tipo de partido (Pilar C). Competitivo > amistoso.
# ---------------------------------------------------------------------------
DEFAULT_MATCH_WEIGHT = 1.2
FRIENDLY_WEIGHT = 0.8
# El Mundial pesa más que el resto: es la evidencia más pura y reciente sobre
# el nivel real de cada equipo (rivales top, máxima intensidad, sin rotaciones).
WORLD_CUP_WEIGHT = 1.5

# Decaimiento temporal: vida media en días. Un partido de hace HALF_LIFE_DAYS
# pesa la mitad que uno de hoy. Solo se usa si hay columna de fecha.
HALF_LIFE_DAYS = 540  # ~18 meses

# Regularización del modelo de ratings (fuerza del prior). Más alto = más
# encogimiento hacia la media (más robusto, menos extremo). Equivale a una
# desviación estándar a priori de ~1/sqrt(RIDGE) en escala log.
#
# NOTA: con RIDGE=8 las lambdas quedaban demasiado comprimidas (todos los
# equipos ~1.3 goles, así que el marcador modal salía 1-1 en el 98% de los
# cruces). Con ~13 partidos por equipo, un prior algo más suave (RIDGE≈4,
# sd a priori ~0.5 en log) abre el abanico de goles a un rango realista
# (~0.6 a ~3.0) SIN reintroducir el sesgo de calendario (Nueva Zelanda sigue
# corregida hacia abajo). Si reconstruyes la base con muchos más partidos por
# equipo, puedes bajarlo aún más.
RIDGE = 4.0


def _match_type_weight(tipo: str) -> float:
    if not isinstance(tipo, str):
        return DEFAULT_MATCH_WEIGHT
    t = tipo.strip().lower()
    if "amistoso" in t:
        return FRIENDLY_WEIGHT
    if "mundial" in t and "clasif" not in t:  # Copa del Mundo, NO 'Clasif. Mundial'
        return WORLD_CUP_WEIGHT
    return DEFAULT_MATCH_WEIGHT


# Grupos de grafías equivalentes para un mismo equipo. NO se privilegia ninguna
# como canónica: al cargar, el canónico será SIEMPRE la grafía que aparezca en
# tu base de datos (columna 'País Analizado'). Así, da igual si la base usa
# 'República Checa' y el ranking FIFA usa 'Rep. Checa': ambas resuelven al
# nombre real de los datos y nada se rompe.
ALIAS_GROUPS = [
    ["Rep. Checa", "República Checa", "Republica Checa", "Chequia", "Czechia"],
    ["Bosnia y Herz.", "Bosnia y Herzegovina", "Bosnia"],
]


def _strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).strip().lower()


@dataclass
class TeamMetrics:
    """Métricas descriptivas + ratings ajustados por rival."""
    team: str
    matches_played: int
    avg_scored: float           # goles a favor ponderados (descriptivo)
    avg_conceded: float         # goles en contra ponderados (descriptivo)
    attack_strength: float      # exp(atk): >1 mejor ataque que la media
    defense_strength: float     # exp(-def): <1 mejor defensa que la media (menos goles recibidos)
    attack_log: float           # atk en escala log (parámetro crudo del modelo)
    defense_log: float          # def en escala log
    clean_sheet_rate: float     # proporción de arcos en cero


class DataManager:
    """Carga la base, normaliza nombres y ajusta los ratings Poisson."""

    REQUIRED_COLUMNS = {
        "País Analizado",
        "Rival (Equipo 2)",
        "Condición",
        "Goles F",
        "Goles C",
        "Tipo de Partido",
        "Arco en Cero",
    }

    # Posibles nombres de columna de fecha (opcional).
    DATE_COLUMNS = ("Fecha", "fecha", "Date", "date", "FECHA")

    def __init__(
        self,
        database_path: str,
        sheet_name: str | int = "Hoja2",
        min_matches: int = 3,
        ridge: float = RIDGE,
        half_life_days: float = HALF_LIFE_DAYS,
        fifa_weight: float = 0.55,
        use_fifa_prior: bool = True,
        style_shrink: float = 0.3,
        max_date: str | None = None,
    ):
        self.database_path = database_path
        self.min_matches = min_matches
        self.ridge = ridge
        self.half_life_days = half_life_days
        self.fifa_weight = float(fifa_weight)
        self.use_fifa_prior = bool(use_fifa_prior)
        self.style_shrink = float(style_shrink)
        # Para backtesting: si se da max_date, el modelo se entrena SOLO con
        # partidos hasta esa fecha (evita fugas de información del futuro).
        self.max_date = pd.to_datetime(max_date) if max_date is not None else None

        self.df = self._load(database_path, sheet_name)

        # Mapa alias -> nombre canónico. Regla: el canónico es la grafía que
        # EXISTE en los datos. Primero registramos los nombres de la base como
        # canónicos; luego, cada grupo de variantes apunta a su miembro presente
        # en los datos (si lo hay).
        self._alias_lookup: dict[str, str] = {}
        data_names = list(self.df["País Analizado"].unique())
        for name in data_names:
            self._alias_lookup[_strip_accents(name)] = name
        data_norm = {_strip_accents(n): n for n in data_names}
        for group in ALIAS_GROUPS:
            canon = next(
                (data_norm[_strip_accents(g)] for g in group
                 if _strip_accents(g) in data_norm),
                group[0],
            )
            for g in group:
                self._alias_lookup[_strip_accents(g)] = canon

        # Promedios globales (descriptivos / compatibilidad).
        w = self.df["_peso"].to_numpy()
        self.global_attack_avg = float(np.average(self.df["Goles F"], weights=w))
        self.global_defense_avg = float(np.average(self.df["Goles C"], weights=w))

        # --- AJUSTE DEL MODELO DE RATINGS (el cambio importante) ---
        self._fit_ratings()
        self._metrics_cache: dict[str, TeamMetrics] = {}

    # ------------------------------------------------------------------ carga
    def _load(self, path: str, sheet_name) -> pd.DataFrame:
        if str(path).lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, sheet_name=sheet_name)

        df.columns = df.columns.str.strip()

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                "La hoja de datos no contiene las columnas requeridas: "
                f"{sorted(missing)}. ¿Estás apuntando a la hoja correcta "
                "(la de partidos, no la de grupos)?"
            )

        for col in ("Goles F", "Goles C", "Arco en Cero"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # Peso por tipo de partido.
        df["_peso"] = df["Tipo de Partido"].apply(_match_type_weight)

        # Decaimiento temporal opcional (si hay columna de fecha).
        date_col = next((c for c in self.DATE_COLUMNS if c in df.columns), None)
        if date_col is not None:
            fechas = pd.to_datetime(df[date_col], errors="coerce")

            # Corte temporal para backtesting: descarta partidos posteriores.
            if self.max_date is not None:
                mask = fechas.notna() & (fechas <= self.max_date)
                df = df.loc[mask].reset_index(drop=True)
                fechas = fechas.loc[mask].reset_index(drop=True)
                if df.empty:
                    raise ValueError(
                        f"No quedan partidos antes de {self.max_date.date()}."
                    )

            if fechas.notna().any():
                ref = fechas.max()
                antig = (ref - fechas).dt.days.fillna(0).clip(lower=0)
                decay = np.power(0.5, antig / float(self.half_life_days))
                df["_peso"] = df["_peso"] * decay
                self._has_dates = True
            else:
                self._has_dates = False
        else:
            self._has_dates = False

        return df

    # -------------------------------------------------------- ratings (MLE)
    def _fit_ratings(self) -> None:
        """
        Ajusta mu, h (localía), y atk/def por equipo maximizando la
        verosimilitud Poisson ponderada con regularización L2.

        Cada fila aporta DOS observaciones de goles:
          1) País Analizado anota 'Goles F' al Rival.
          2) Rival anota 'Goles C' al País Analizado.
        """
        df = self.df

        # Universo de equipos: clasificados + rivales (en su forma canónica
        # cuando aplica; para rivales no clasificados, su nombre tal cual).
        teams = pd.unique(
            pd.concat([df["País Analizado"], df["Rival (Equipo 2)"]])
        ).tolist()
        self._team_index = {t: i for i, t in enumerate(teams)}
        self._teams = teams
        T = len(teams)

        pais = df["País Analizado"].map(self._team_index).to_numpy()
        rival = df["Rival (Equipo 2)"].map(self._team_index).to_numpy()
        gf = df["Goles F"].to_numpy(float)
        gc = df["Goles C"].to_numpy(float)
        peso = df["_peso"].to_numpy(float)
        cond = df["Condición"].astype(str).str.strip().str.lower().to_numpy()

        # local_pais = 1 si el País Analizado jugó de local; -1 si de visitante; 0 neutral.
        # (la localía del rival es el signo opuesto)
        loc = np.zeros(len(df))
        loc[cond == "local"] = 1.0
        loc[cond == "visitante"] = -1.0

        # Dos bloques de observaciones apilados:
        #   bloque 1 (scorer=pais, conceder=rival, home_sign = loc)
        #   bloque 2 (scorer=rival, conceder=pais, home_sign = -loc)
        scorer = np.concatenate([pais, rival])
        conceder = np.concatenate([rival, pais])
        y = np.concatenate([gf, gc])
        w = np.concatenate([peso, peso])
        home = np.concatenate([loc, -loc])  # +1 si el scorer fue local

        # Parámetros: [mu, h, atk(0..T-1), def(0..T-1)]
        n_param = 2 + 2 * T

        def unpack(theta):
            mu = theta[0]
            h = theta[1]
            atk = theta[2:2 + T]
            dfn = theta[2 + T:]
            return mu, h, atk, dfn

        ridge = self.ridge

        def nll_and_grad(theta):
            mu, h, atk, dfn = unpack(theta)
            eta = mu + atk[scorer] - dfn[conceder] + h * home
            eta = np.clip(eta, -10, 10)
            lam = np.exp(eta)
            resid = w * (lam - y)          # dNLL/deta por observación

            # NLL Poisson ponderada (sin el log(y!) constante) + ridge.
            nll = np.sum(w * (lam - y * eta))
            nll += 0.5 * ridge * (np.sum(atk ** 2) + np.sum(dfn ** 2))

            g = np.zeros(n_param)
            g[0] = resid.sum()                       # mu
            g[1] = (resid * home).sum()              # h
            np.add.at(g, 2 + scorer, resid)          # atk del scorer
            np.add.at(g, 2 + T + conceder, -resid)   # def del conceder (signo -)
            g[2:2 + T] += ridge * atk
            g[2 + T:] += ridge * dfn
            return nll, g

        from scipy.optimize import minimize

        theta0 = np.zeros(n_param)
        res = minimize(
            nll_and_grad, theta0, jac=True, method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
        )
        mu, h, atk, dfn = unpack(res.x)

        # Centramos atk y def (suma cero) trasladando la masa a mu, para que
        # exp(atk)=1 signifique "ataque promedio". No cambia las predicciones.
        atk_mean = atk.mean()
        dfn_mean = dfn.mean()
        mu_centered = mu + atk_mean - dfn_mean
        atk = atk - atk_mean
        dfn = dfn - dfn_mean

        self._mu = float(mu_centered)
        self._home_adv = float(h)
        self._atk = atk
        self._def = dfn
        self._fit_success = bool(res.success)

        # --- Mezcla con el prior FIFA (ancla de fuerza del calendario) ---
        if self.use_fifa_prior:
            self._apply_fifa_prior()

    def _apply_fifa_prior(self) -> None:
        """
        Corrige el sesgo por fuerza del calendario anclando la FUERZA de cada
        equipo al ranking FIFA. Clave conceptual: en el modelo los goles de i
        a j son exp(mu + atk_i - def_j), con def ALTO = mejor defensa. Por tanto
        la fuerza global de un equipo es atk + def (mete mucho Y recibe poco),
        NO atk - def (eso es solo su ESTILO ofensivo/defensivo).

        Los datos no distinguen 'fuerte' de 'goleó a rivales débiles': Japón o
        Nueva Zelanda acumulan atk alto y def alta jugando en zonas flojas, lo
        que infla su fuerza total. El ranking FIFA sí carga la calidad real del
        calendario, así que lo usamos como ancla.

        Procedimiento (en la escala log del modelo):
          1) strength_i = atk_i + def_i ; style_i = atk_i - def_i (48 clasif.).
          2) Ancla la FUERZA al FIFA: OLS strength ~ a + b*FIFA (con respaldo
             z-score si la relación degenera por el propio sesgo de los datos).
          3) strength_final = (1-w)*strength_data + w*strength_fifa.
          4) El ESTILO también está sesgado por el calendario -> se encoge
             hacia 0 con style_shrink (estabiliza ataques/defensas extremos).
          5) Reconstruye atk=(S+style)/2, def=(S-style)/2: coherente y sin
             dobles cuentas.
        """
        from fifa_prior import fifa_points

        qualified = self.list_teams()
        idxs, fifa_pts, strength_data, style_data = [], [], [], []
        for t in qualified:
            pts = fifa_points(t)
            if pts is None:
                continue
            i = self._team_index[t]
            idxs.append(i)
            fifa_pts.append(pts)
            strength_data.append(self._atk[i] + self._def[i])
            style_data.append(self._atk[i] - self._def[i])

        if len(idxs) < 5:
            return  # sin prior suficiente, dejamos los ratings de datos

        idxs = np.array(idxs)
        fifa_pts = np.array(fifa_pts, float)
        strength_data = np.array(strength_data, float)
        style_data = np.array(style_data, float)

        # Ancla la FUERZA al FIFA. OLS strength ~ a + b*fifa; respaldo z-score.
        X = np.column_stack([np.ones_like(fifa_pts), fifa_pts])
        coef, *_ = np.linalg.lstsq(X, strength_data, rcond=None)
        a, b = coef
        if b <= 0:  # relación degenerada -> usa el orden por z-score de FIFA
            mu_f, sd_f = fifa_pts.mean(), fifa_pts.std() + 1e-9
            strength_fifa = (
                strength_data.mean()
                + strength_data.std() * (fifa_pts - mu_f) / sd_f
            )
        else:
            strength_fifa = a + b * fifa_pts

        w = self.fifa_weight
        strength_final = (1 - w) * strength_data + w * strength_fifa
        style_final = (1.0 - self.style_shrink) * style_data  # encoge el estilo

        # Reconstruye atk y def de forma consistente.
        self._atk[idxs] = (strength_final + style_final) / 2.0
        self._def[idxs] = (strength_final - style_final) / 2.0
        self._fifa_blend_info = {
            "slope": float(b), "intercept": float(a), "weight": w,
            "style_shrink": self.style_shrink, "n_anchored": len(idxs),
            "fallback_zscore": bool(b <= 0),
        }

    # ----------------------------------------------------------- nombres
    def resolve_name(self, name: str) -> str:
        key = _strip_accents(name)
        if key in self._alias_lookup:
            return self._alias_lookup[key]
        raise ValueError(
            f"El equipo '{name}' no está en la base de datos. "
            f"Equipos disponibles: {self.list_teams()[:5]}..."
        )

    def list_teams(self) -> list[str]:
        """Equipos CLASIFICADOS (los analizados), ordenados."""
        return sorted(self.df["País Analizado"].unique())

    # ----------------------------------------------------- ratings públicos
    def _team_rating(self, canonical: str) -> tuple[float, float]:
        """(atk_log, def_log) de un equipo en escala del modelo."""
        idx = self._team_index[canonical]
        return float(self._atk[idx]), float(self._def[idx])

    @property
    def home_advantage(self) -> float:
        """Ventaja de localía estimada de los datos (en escala log)."""
        return self._home_adv

    def base_lambdas(self, team_a: str, team_b: str, venue: str = "neutral"):
        """
        Goles esperados (lambda) de cada equipo según el modelo de ratings
        ajustado por rival. Incluye localía estimada de los datos.

        venue: "neutral" | "home_a" | "home_b"
        """
        a = self.resolve_name(team_a)
        b = self.resolve_name(team_b)
        atk_a, def_a = self._team_rating(a)
        atk_b, def_b = self._team_rating(b)

        home_a = 1.0 if venue == "home_a" else 0.0
        home_b = 1.0 if venue == "home_b" else 0.0

        lam_a = np.exp(self._mu + atk_a - def_b + self._home_adv * home_a)
        lam_b = np.exp(self._mu + atk_b - def_a + self._home_adv * home_b)
        return float(lam_a), float(lam_b)

    # --------------------------------------------------------- métricas
    def get_team_metrics(self, team: str) -> TeamMetrics:
        canonical = self.resolve_name(team)
        if canonical in self._metrics_cache:
            return self._metrics_cache[canonical]

        data = self.df[self.df["País Analizado"] == canonical]
        if data.empty:
            raise ValueError(f"Sin datos para '{canonical}'.")

        w = data["_peso"].to_numpy()
        avg_scored = float(np.average(data["Goles F"], weights=w))
        avg_conceded = float(np.average(data["Goles C"], weights=w))
        atk_log, def_log = self._team_rating(canonical)
        clean_sheet_rate = float(data["Arco en Cero"].mean())

        m = TeamMetrics(
            team=canonical,
            matches_played=len(data),
            avg_scored=avg_scored,
            avg_conceded=avg_conceded,
            attack_strength=float(np.exp(atk_log)),
            defense_strength=float(np.exp(-def_log)),  # <1 = defensa mejor que media
            attack_log=atk_log,
            defense_log=def_log,
            clean_sheet_rate=clean_sheet_rate,
        )
        self._metrics_cache[canonical] = m
        return m

    def data_quality_warnings(self) -> list[str]:
        warnings = []
        counts = self.df["País Analizado"].value_counts()
        for team, n in counts.items():
            if n < self.min_matches:
                warnings.append(f"'{team}' solo tiene {n} partidos (poco fiable).")
        return warnings

    def ratings_table(self) -> pd.DataFrame:
        """Tabla ordenada de ratings de los equipos clasificados (para inspección)."""
        rows = []
        for t in self.list_teams():
            m = self.get_team_metrics(t)
            lam_vs_avg, _ = self.base_lambdas(t, t)  # no usado; placeholder
            rows.append({
                "Equipo": m.team,
                "ATK": round(m.attack_strength, 3),
                "DEF": round(m.defense_strength, 3),
                "Rating": round(m.attack_log + m.defense_log, 3),  # FUERZA total
                "Estilo": round(m.attack_log - m.defense_log, 3),  # + ofensivo / - defensivo
                "GF_obs": round(m.avg_scored, 2),
                "GC_obs": round(m.avg_conceded, 2),
                "CS%": round(m.clean_sheet_rate * 100, 0),
            })
        return pd.DataFrame(rows).sort_values("Rating", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    dm = DataManager("base_mundial_2026.xlsx")
    print(f"Equipos clasificados: {len(dm.list_teams())}")
    print(f"Universo de ratings (incluye rivales): {len(dm._teams)}")
    print(f"mu={dm._mu:.3f}  ventaja_localia(log)={dm.home_advantage:.3f} "
          f"(x{np.exp(dm.home_advantage):.2f} goles)  fit_ok={dm._fit_success}")
    print()
    tbl = dm.ratings_table()
    print("TOP 10 (rating = fuerza total = atk + def):")
    print(tbl.head(10).to_string(index=False))
    print("\nBOTTOM 8:")
    print(tbl.tail(8).to_string(index=False))
