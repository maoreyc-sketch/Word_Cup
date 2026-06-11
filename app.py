"""
app.py
======
Interfaz Streamlit del Motor de Predicción del Mundial 2026.

Tres vistas:
  1) Predicción de un partido (Poisson + Dixon-Coles, probabilidades exactas).
  2) Tabla de fuerzas de los 48 clasificados (ratings ajustados por rival
     y anclados al ranking FIFA, ya sin sesgo de fuerza-de-calendario).
  3) Simulación Montecarlo del torneo COMPLETO (grupos + eliminatorias).

Parámetros por defecto = los que sugiere calibration.py sobre tu base actual.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_manager import DataManager
from predictive_engine import StatisticalPredictor
from tournament_simulator import TournamentSimulator, HOSTS
from app_groups import GRUPOS

st.set_page_config(
    page_title="World Cup 2026 - Predictive Engine",
    page_icon="⚽",
    layout="wide",
)

# Valores calibrados a la base (clasificado vs clasificado) por calibration.py:
# total ~2.59 goles, 0-0 ~5%. La base NO necesita inflar empates (rho≈0):
# el rho negativo anterior (-0.12) sobreestimaba el 1-1 y lo dejaba pegado como
# marcador modal. calibration.py los re-estima si cambias la base.
CALIB_GOAL_SCALING = 1.079
CALIB_RHO = 0.0

st.title("📊 Motor de Predicción Estocástica · Mundial 2026")
st.caption("Poisson + Dixon-Coles · ratings ajustados por rival y anclados al ranking FIFA")
st.markdown("---")


@st.cache_resource
def build_dm(fifa_weight: float, half_life_days: float):
    return DataManager(
        "base_mundial_2026.xlsx",
        fifa_weight=fifa_weight,
        half_life_days=half_life_days,
    )


try:
    # DM base con valores por defecto: sirve para leer los defaults y para
    # detectar si la base trae fechas (decaimiento temporal).
    base_dm = build_dm(0.55, 540.0)
except Exception as e:
    st.error(
        "No pude cargar la base de datos. Verifica que 'base_mundial_2026.xlsx' "
        f"esté en la misma carpeta y que la hoja de partidos exista.\n\nError: {e}"
    )
    st.stop()

# ==========================================
# Panel lateral: calibración
# ==========================================
st.sidebar.header("⚙️ Calibración del modelo")
st.sidebar.caption(
    "Los ratings ya están ajustados por la fuerza del rival y mezclados con el "
    "ranking FIFA, así que el nivel de goles está bien escalado de fábrica. "
    "Los valores por defecto provienen de calibration.py sobre tu base."
)
goal_scaling = st.sidebar.slider(
    "Calibración de goles (fiel a datos)", 0.50, 1.30, CALIB_GOAL_SCALING, 0.01,
    help="Ajusta el total de goles al observado entre clasificados. 1.076 = calibrado.",
)
goal_environment = st.sidebar.slider(
    "Ambiente goleador (creencia)", 0.70, 1.50, 1.00, 0.05,
    help=">1.0 = crees que el Mundial será más abierto que tus registros.",
)
rho = st.sidebar.slider(
    "Corrección de empates (Dixon-Coles ρ)", -0.20, 0.00, CALIB_RHO, 0.01,
    help="Más negativo = más empates y marcadores bajos. Tus datos sugieren -0.12.",
)
fifa_w = st.sidebar.slider(
    "Peso del ranking FIFA (ancla)", 0.0, 0.9, float(base_dm.fifa_weight), 0.05,
    help="0 = solo tus datos (con su sesgo de calendario). 0.55 = mezcla recomendada.",
)

# --- Decaimiento temporal (time-decay) ---
st.sidebar.markdown("---")
st.sidebar.subheader("🕒 Decaimiento temporal")
if base_dm._has_dates:
    st.sidebar.caption(
        "La base trae columna de fecha: los partidos recientes pesan más "
        "(decaimiento exponencial). Vida media = días en los que un partido "
        "pasa a pesar la mitad."
    )
    half_life = st.sidebar.slider(
        "Vida media (días)", 90, 1095, 540, 30,
        help="Más bajo = solo cuenta el presente (forma reciente). "
             "Más alto = la historia pesa más. 540 ≈ 18 meses.",
    )
else:
    st.sidebar.info(
        "La base no tiene columna de fecha utilizable, así que el decaimiento "
        "temporal está desactivado (todos los partidos pesan igual)."
    )
    half_life = 540

# Reconstruimos el DataManager con los parámetros elegidos (cacheado).
data_manager = build_dm(round(fifa_w, 2), float(half_life))

predictor = StatisticalPredictor(
    data_manager, rho=rho, goal_scaling=goal_scaling, goal_environment=goal_environment
)

for w in predictor.data.data_quality_warnings():
    st.sidebar.warning(w)

tab_match, tab_ratings, tab_tourney, tab_valid = st.tabs(
    ["🎯 Predicción de partido", "📋 Tabla de fuerzas", "🏆 Simular torneo",
     "📈 Validación (backtest)"]
)

# ==========================================
# TAB 1 · Predicción de un partido
# ==========================================
with tab_match:
    st.caption(
        "Selecciona **cualquier** par de selecciones — sirve para simular cruces "
        "de eliminatoria que no comparten grupo (p. ej. una hipotética final)."
    )
    # Lista plana de los 48, para cruces libres.
    ALL_TEAMS = sorted({t for teams in GRUPOS.values() for t in teams})

    modo = st.radio(
        "Modo de selección",
        ["Cruce libre (cualquier combinación)", "Filtrar por grupo"],
        horizontal=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if modo == "Filtrar por grupo":
            grupo = st.selectbox("Grupo", list(GRUPOS.keys()),
                                 format_func=lambda g: f"Grupo {g}")
            pool = GRUPOS[grupo]
        else:
            pool = ALL_TEAMS
        venue = st.radio(
            "Sede del partido",
            ["Neutral (Mundial)", "Ventaja Equipo 1", "Ventaja Equipo 2"],
            horizontal=True,
        )
    with col2:
        equipo_a = st.selectbox("1️⃣ Equipo 1", pool, index=0)
        equipo_b = st.selectbox(
            "2️⃣ Equipo 2", [e for e in pool if e != equipo_a], index=0
        )

    venue_map = {"Neutral (Mundial)": "neutral", "Ventaja Equipo 1": "home_a", "Ventaja Equipo 2": "home_b"}

    if st.button("🚀 CALCULAR PROBABILIDADES", type="primary"):
        with st.spinner("Calculando distribución de marcadores..."):
            pred = predictor.predict_match(equipo_a, equipo_b, venue=venue_map[venue])
            r = pred.as_dict()

        st.success(f"Predicción · {pred.team_a} vs {pred.team_b}")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Victoria {pred.team_a}", f"{r['prob_local']}%")
        c2.metric("Empate 🤝", f"{r['prob_empate']}%")
        c3.metric(f"Victoria {pred.team_b}", f"{r['prob_visitante']}%")
        st.caption(
            f"Goles esperados (λ): {pred.team_a} = {r['lambda_local']} · "
            f"{pred.team_b} = {r['lambda_visitante']}"
        )
        st.markdown("---")

        col_izq, col_der = st.columns(2)
        with col_izq:
            st.subheader("📊 Resultado del partido")
            fig = go.Figure([go.Bar(
                x=[pred.team_a, "Empate", pred.team_b],
                y=[r["prob_local"], r["prob_empate"], r["prob_visitante"]],
                marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
            )])
            fig.update_layout(yaxis_title="Probabilidad (%)", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        with col_der:
            st.subheader("🎯 Mercados avanzados")
            st.markdown(f"**Marcador más probable:** `{r['marcador_exacto'][0]} - {r['marcador_exacto'][1]}`")
            st.markdown(f"**Over 2.5 goles:** {r['prob_over_2_5']}%")
            st.markdown(f"**Ambos marcan:** {r['prob_ambos_marcan']}%")
            st.progress(min(int(r["prob_over_2_5"]), 100))

        st.subheader("🔥 Probabilidad de cada marcador exacto")
        m = pred.score_matrix[:6, :6] * 100
        heat = go.Figure(data=go.Heatmap(
            z=m,
            x=[f"{pred.team_b}: {i}" for i in range(6)],
            y=[f"{pred.team_a}: {i}" for i in range(6)],
            colorscale="Viridis", text=np.round(m, 1), texttemplate="%{text}%",
        ))
        heat.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(heat, use_container_width=True)

# ==========================================
# TAB 2 · Tabla de fuerzas
# ==========================================
with tab_ratings:
    st.subheader("📋 Fuerza de los 48 clasificados")
    st.caption(
        "Rating = fuerza total (ataque + defensa, escala log del modelo). "
        "Estilo > 0 = más ofensivo; < 0 = más defensivo. Ya corregido por la "
        "calidad del rival y anclado al ranking FIFA."
    )
    tbl = data_manager.ratings_table()
    st.dataframe(tbl, use_container_width=True, height=560)
    fig = go.Figure([go.Bar(
        x=tbl["Rating"][:20][::-1], y=tbl["Equipo"][:20][::-1],
        orientation="h", marker_color="#1f77b4",
    )])
    fig.update_layout(template="plotly_dark", height=560, xaxis_title="Rating (fuerza)")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 3 · Simulación del torneo
# ==========================================
with tab_tourney:
    st.subheader("🏆 Montecarlo del Mundial completo")
    st.caption(
        "Grupos (todos contra todos) + mejores terceros + eliminatorias con "
        "el cuadro oficial 2026. Cada eliminatoria incluye prórroga/penales "
        "modelados por la fuerza relativa."
    )
    c1, c2, c3 = st.columns(3)
    n_sims = c1.select_slider("N° de simulaciones", [2000, 5000, 10000, 20000, 50000], value=10000)
    seed = c2.number_input("Semilla", value=42, step=1)
    use_host = c3.checkbox("Ventaja anfitrión (México/EEUU/Canadá)", value=True)

    if st.button("🎲 SIMULAR TORNEO", type="primary"):
        with st.spinner(f"Corriendo {n_sims:,} torneos completos..."):
            sim = TournamentSimulator(predictor, GRUPOS, host_advantage=use_host)
            res = sim.run(N=int(n_sims), seed=int(seed))
        df = pd.DataFrame([
            {"Equipo": t, "Pasa de grupo %": v["grupo"], "R16 %": v["r16"],
             "Cuartos %": v["qf"], "Semis %": v["sf"], "Final %": v["final"],
             "CAMPEÓN %": v["campeon"]}
            for t, v in res.items()
        ]).sort_values("CAMPEÓN %", ascending=False).reset_index(drop=True)

        st.dataframe(df, use_container_width=True, height=500)
        top = df.head(15)
        fig = go.Figure([go.Bar(
            x=top["CAMPEÓN %"][::-1], y=top["Equipo"][::-1],
            orientation="h", marker_color="#2ca02c",
        )])
        fig.update_layout(template="plotly_dark", height=520,
                          xaxis_title="Probabilidad de ser campeón (%)")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 4 · Validación fuera-de-muestra
# ==========================================
with tab_valid:
    st.subheader("📈 ¿Qué tan confiables son las probabilidades?")
    st.caption(
        "Backtest con ventana expansiva: el modelo se entrena solo con partidos "
        "ANTERIORES a cada fecha de corte y predice los siguientes (sin ver el "
        "futuro). Las probabilidades se puntúan con reglas propias (RPS, Brier, "
        "log-loss) y se comparan contra dos baselines. Menor = mejor."
    )

    @st.cache_resource
    def cached_backtest(fifa_w_bt: float):
        from backtest import run_backtest
        return run_backtest(fifa_weight=fifa_w_bt)

    if st.button("🔬 CORRER BACKTEST", type="primary"):
        with st.spinner("Re-entrenando el modelo en cada corte temporal..."):
            try:
                bt = cached_backtest(round(fifa_w, 2))
            except Exception as e:
                st.error(f"No se pudo correr el backtest: {e}")
                st.stop()

        s = bt["summary"]
        st.markdown(f"**{s['n_predicciones']} predicciones fuera-de-muestra** "
                    f"en {s['folds']} cortes temporales.")

        c1, c2, c3 = st.columns(3)
        c1.metric("RPS 1X2 (modelo)", f"{s['rps_modelo']:.4f}",
                  delta=f"{s['rps_uniforme'] - s['rps_modelo']:+.4f} vs azar",
                  delta_color="normal")
        c2.metric("Brier 1X2 (modelo)", f"{s['brier_modelo']:.4f}",
                  delta=f"{s['brier_uniforme'] - s['brier_modelo']:+.4f} vs azar",
                  delta_color="normal")
        c3.metric("Acierto duro 1X2", f"{s['acierto_1x2']}%",
                  help="Frecuencia con la que el resultado más probable fue el real. "
                       "33% sería el azar puro en 3 salidas.")

        st.markdown("---")
        st.markdown("##### Mercados de goles")
        g1, g2, g3 = st.columns(3)
        g1.metric("Brier Over 2.5", f"{s['brier_over25_modelo']:.4f}",
                  delta=f"{s['brier_over25_50pct'] - s['brier_over25_modelo']:+.4f} "
                        "vs apostar 50% siempre",
                  delta_color="normal")
        g2.metric("Brier Ambos marcan", f"{s['brier_btts_modelo']:.4f}")
        g3.metric("Goles/partido (pred vs real)",
                  f"{s['goles_predichos_media']} / {s['goles_reales_media']}")

        st.markdown("##### Curva de calibración · Over 2.5")
        st.caption(
            "Un modelo calibrado cae sobre la diagonal: cuando dice 60%, "
            "acierta ~60% de esas veces. Desviaciones grandes = sobre/sub-confianza."
        )
        calib = bt["calibracion_over25"]
        calib_plot = calib[calib["n"] > 0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 name="Calibración perfecta",
                                 line=dict(dash="dash", color="#888")))
        fig.add_trace(go.Scatter(
            x=calib_plot["p_media"], y=calib_plot["freq_real"],
            mode="markers+lines", name="Modelo",
            marker=dict(size=calib_plot["n"].clip(6, 30), color="#2ca02c"),
            text=[f"n={int(n)}" for n in calib_plot["n"]],
        ))
        fig.update_layout(template="plotly_dark", height=420,
                          xaxis_title="Probabilidad pronosticada de Over 2.5",
                          yaxis_title="Frecuencia real",
                          xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Ver predicciones individuales fuera-de-muestra"):
            det = bt["detalle"][[
                "fecha", "equipo_a", "equipo_b", "marcador",
                "p_win_a", "p_draw", "p_win_b", "p_over25", "over25_real",
            ]].copy()
            for c in ("p_win_a", "p_draw", "p_win_b", "p_over25"):
                det[c] = (det[c] * 100).round(1)
            st.dataframe(det, use_container_width=True, height=400)
