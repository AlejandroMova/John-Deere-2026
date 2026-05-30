import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

JD_GREEN       = "#367C2B"
JD_YELLOW      = "#FFDE00"
DARK_BG        = "#161616"
SURFACE        = "#212121"
SURFACE_HIGH   = "#2C2C2C"
BORDER         = "#333333"
TEXT_PRIMARY   = "#F0F0F0"
TEXT_MUTED     = "#888888"

st.set_page_config(
    page_title="AgroIntel — John Deere",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
  .stApp {{ background-color: {DARK_BG}; }}
  .main .block-container {{ padding-top: 2rem; max-width: 1400px; }}

  /* Reset Streamlit defaults */
  h1, h2, h3, h4, p, label, div {{ color: {TEXT_PRIMARY}; }}

  /* Sliders */
  [data-testid="stSlider"] {{ padding: 0.1rem 0 0.4rem 0; }}
  .stSlider label p {{ color: {TEXT_MUTED} !important; font-size: 0.82rem !important; }}

  /* Metric cards */
  [data-testid="metric-container"] {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 1rem 1.1rem;
  }}
  [data-testid="stMetricValue"] {{
    color: {TEXT_PRIMARY} !important;
    font-size: 1.7rem !important;
    font-weight: 600 !important;
  }}
  [data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED} !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  [data-testid="stMetricDelta"] {{
    font-size: 0.85rem !important;
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background-color: transparent;
    border-bottom: 1px solid {BORDER};
    gap: 0;
    padding: 0;
  }}
  .stTabs [data-baseweb="tab"] {{
    color: {TEXT_MUTED};
    font-size: 0.88rem;
    font-weight: 500;
    padding: 10px 20px;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
  }}
  .stTabs [aria-selected="true"] {{
    color: {TEXT_PRIMARY} !important;
    background-color: transparent !important;
    border-bottom: 2px solid {JD_GREEN} !important;
  }}
  .stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.5rem; }}

  /* Dividers */
  hr {{ border: none; border-top: 1px solid {BORDER}; margin: 1.5rem 0; }}
</style>
""", unsafe_allow_html=True)


# ─── Hardcoded data ───────────────────────────────────────────────────────────
SIMILAR_HARVESTS = pd.DataFrame({
    "humedad_grano":    [14.2, 15.8, 16.1, 13.9, 17.2, 15.0, 18.3, 14.7, 16.9, 15.5,
                         17.8, 13.5, 16.4, 14.0, 18.1, 15.3, 17.0, 16.7, 14.5, 15.9],
    "rendimiento_real": [7.2,  6.8,  6.5,  7.5,  6.1,  7.0,  5.9,  7.3,  6.3,  6.9,
                         6.0,  7.6,  6.6,  7.4,  5.8,  7.1,  6.2,  6.4,  7.2,  6.7],
    "distance":         [0.05, 0.09, 0.12, 0.06, 0.18, 0.10, 0.22, 0.08, 0.15, 0.11,
                         0.20, 0.07, 0.13, 0.06, 0.23, 0.10, 0.17, 0.14, 0.09, 0.12],
})

HIST_RENDIMIENTOS = pd.DataFrame({"rendimiento_real": [
    5.2, 6.8, 7.1, 4.9, 6.3, 7.5, 5.8, 6.1, 7.8, 5.5,
    6.6, 7.2, 5.0, 6.9, 7.4, 5.3, 6.4, 7.0, 5.7, 6.2,
    7.6, 5.1, 6.7, 7.3, 5.4, 6.0, 7.9, 5.6, 6.5, 7.1,
    4.8, 6.8, 7.5, 5.2, 6.3, 7.2, 5.9, 6.1, 7.7, 5.4,
    6.6, 7.0, 5.1, 6.9, 7.4, 5.7, 6.4, 7.3, 5.0, 6.2,
    7.6, 5.3, 6.7, 7.1, 5.5, 6.0, 7.8, 5.8, 6.5, 7.2,
    4.9, 6.8, 7.5, 5.2, 6.3, 7.0, 5.9, 6.2, 7.7, 5.4,
    6.6, 7.3, 5.1, 6.9, 7.4, 5.7, 6.4, 7.1, 5.0, 6.1,
    7.6, 5.3, 6.7, 7.2, 5.5, 6.0, 7.9, 5.8, 6.5, 7.0,
    4.8, 6.8, 7.5, 5.2, 6.3, 7.3, 5.9, 6.2, 7.7, 5.4,
    6.6, 7.1, 5.1, 6.9, 7.4, 5.7, 6.4, 7.2, 5.0, 6.2,
    7.6, 5.3, 6.7, 7.0, 5.5, 6.0, 7.8, 5.8, 6.5, 7.3,
    4.9, 6.8, 7.5, 5.2, 6.3, 7.1, 5.9, 6.1, 7.7, 5.4,
    6.6, 7.2, 5.1, 6.9, 7.4, 5.7, 6.4, 7.0, 5.0, 6.2,
    7.6, 5.3, 6.7, 7.3, 5.5, 6.0, 7.9, 5.8, 6.5, 7.1,
    4.8, 6.8, 7.5, 5.2, 6.3, 7.2, 5.9, 6.2, 7.7, 5.4,
    6.6, 7.0, 5.1, 6.9, 7.4, 5.7, 6.4, 7.3, 5.0, 6.1,
    7.6, 5.3, 6.7, 7.1, 5.5, 6.0, 7.8, 5.8, 6.5, 7.2,
    4.9, 6.8, 7.5, 5.2, 6.3, 7.0, 5.9, 6.2, 7.7, 5.4,
    6.6, 7.3, 5.1, 6.9, 7.4, 5.7, 6.4, 7.1, 5.0, 6.2,
]})

VAL_DATA = pd.DataFrame({
    "Real":     [7.2, 6.8, 5.5, 7.5, 6.1, 6.9, 5.8, 7.3, 6.3, 7.0,
                 5.9, 7.6, 6.6, 7.4, 5.7, 7.1, 6.2, 6.4, 7.2, 6.7,
                 5.4, 7.0, 6.5, 7.3, 5.8, 6.9, 7.1, 6.3, 5.6, 7.4],
    "Predicho": [7.0, 6.9, 5.7, 7.3, 6.3, 7.1, 6.0, 7.2, 6.1, 6.8,
                 6.1, 7.4, 6.8, 7.2, 5.9, 7.0, 6.4, 6.6, 7.0, 6.9,
                 5.6, 6.8, 6.7, 7.1, 6.0, 7.1, 6.9, 6.5, 5.8, 7.2],
})

# Hardcoded prediction output
PRED_RENDIMIENTO = 6.85
PRED_COSTO       = 682.0
PRED_CONFIANZA   = 78
PRED_DELTA_PCT   = +14.2
REC_LEVEL        = "green"
MAE              = 0.31


# ─── Header ───────────────────────────────────────────────────────────────────
col_logo, col_meta = st.columns([3, 2])
with col_logo:
    st.markdown(
        f'<div style="font-size:1.05rem; color:{TEXT_MUTED}; font-weight:500; margin-bottom:2px;">'
        f'<span style="color:{JD_GREEN}; font-weight:700;">John Deere</span>'
        f' &nbsp;·&nbsp; AgroIntel</div>'
        f'<div style="font-size:1.75rem; font-weight:700; color:{TEXT_PRIMARY}; line-height:1.2;">'
        f'Simulador de Decisiones de Cosecha</div>',
        unsafe_allow_html=True,
    )
with col_meta:
    st.markdown(
        f'<div style="text-align:right; padding-top:0.8rem;">'
        f'<span style="color:{TEXT_MUTED}; font-size:0.8rem;">'
        f'Reto 03 · Datos · Tec de Monterrey</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "Simulador de escenarios",
    "Semáforo operacional",
    "Historial del modelo",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — SIMULADOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_in, col_out = st.columns([1, 1.8], gap="large")

    with col_in:
        st.markdown(
            f'<p style="font-size:0.7rem; color:{TEXT_MUTED}; text-transform:uppercase; '
            f'letter-spacing:1px; margin-bottom:1rem;">Parámetros de entrada</p>',
            unsafe_allow_html=True,
        )

        st.caption("Condiciones del grano")
        st.slider("Humedad del grano (%)", 10, 30, 18)
        st.slider("Rendimiento esperado base (ton/ha)", 2.0, 12.0, 6.0, 0.1,
                  help="Expectativa de rendimiento sin ajustes")

        st.caption("Insumos")
        st.slider("Fertilizante (kg/ha)", 0, 300, 120)
        st.slider("Pesticida (L/ha)", 0.0, 10.0, 2.5, 0.1)

        st.caption("Condiciones ambientales")
        st.slider("Temperatura ambiente (°C)", 5, 45, 28)
        st.slider("Humedad del aire (%)", 20, 95, 60)

        st.caption("Estado de la máquina")
        st.slider("Horas de motor acumuladas", 0, 5000, 1200)
        st.slider("Temperatura del motor (°C)", 60, 120, 85)
        st.slider("Velocidad de cosecha (km/h)", 2.0, 8.0, 5.0, 0.1)

    with col_out:
        st.markdown(
            f'<p style="font-size:0.7rem; color:{TEXT_MUTED}; text-transform:uppercase; '
            f'letter-spacing:1px; margin-bottom:1rem;">Resultados — basados en 20 cosechas similares</p>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rendimiento proyectado", f"{PRED_RENDIMIENTO:.2f} t/ha",
                  delta=f"{PRED_DELTA_PCT:+.1f}% vs base")
        m2.metric("Costo estimado", f"${PRED_COSTO:,.0f}")
        m3.metric("Confianza del modelo", f"{PRED_CONFIANZA}%",
                  help="Similitud promedio de las 20 cosechas de referencia")
        m4.metric("Cosechas de referencia", "20")

        # Status strip
        status_cfg = {
            "green":  (JD_GREEN,  "#1a2e19", "Condiciones favorables — se puede proceder."),
            "yellow": (JD_YELLOW, "#2e2a14", "Rango normal — monitorear humedad del grano."),
            "red":    ("#B94040", "#2e1414", "Por debajo de lo esperado — revisar condiciones."),
        }
        s_color, s_bg, s_text = status_cfg[REC_LEVEL]
        st.markdown(f"""
        <div style="
          background:{s_bg}; border-left:3px solid {s_color};
          padding:0.7rem 1rem; border-radius:0 4px 4px 0;
          margin:1rem 0; font-size:0.88rem; color:{TEXT_PRIMARY};">
          {s_text}
        </div>
        """, unsafe_allow_html=True)

        # Scatter: similar harvests
        opacity_vals = [max(0.35, 1 - d * 3) for d in SIMILAR_HARVESTS["distance"]]
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=SIMILAR_HARVESTS["humedad_grano"],
            y=SIMILAR_HARVESTS["rendimiento_real"],
            mode="markers",
            marker=dict(
                size=9,
                color=[f"rgba(54,124,43,{o:.2f})" for o in opacity_vals],
                line=dict(width=0),
            ),
            name="Cosechas históricas",
            hovertemplate="Humedad: %{x:.1f}%<br>Rendimiento: %{y:.2f} t/ha<extra></extra>",
        ))
        fig_scatter.add_trace(go.Scatter(
            x=[18], y=[PRED_RENDIMIENTO],
            mode="markers",
            marker=dict(symbol="diamond", size=12, color=JD_YELLOW,
                        line=dict(color=DARK_BG, width=1)),
            name="Tu escenario",
        ))
        fig_scatter.update_layout(
            title=dict(text="20 cosechas más similares", font=dict(size=13, color=TEXT_MUTED)),
            xaxis_title="Humedad del grano (%)",
            yaxis_title="Rendimiento (t/ha)",
            template="plotly_dark",
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            margin=dict(l=0, r=0, t=36, b=0),
            legend=dict(font=dict(color=TEXT_MUTED, size=11)),
            xaxis=dict(gridcolor=BORDER),
            yaxis=dict(gridcolor=BORDER),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Impact suggestions
        st.markdown(
            f'<p style="font-size:0.7rem; color:{TEXT_MUTED}; text-transform:uppercase; '
            f'letter-spacing:1px; margin:0.5rem 0 0.75rem 0;">Ajustes sugeridos</p>',
            unsafe_allow_html=True,
        )
        suggestions = [
            ("Humedad del grano",  "18%",      "14%",       "+0.48 t/ha"),
            ("Fertilizante",       "120 kg/ha", "200 kg/ha", "+0.31 t/ha"),
            ("Temperatura motor",  "85 °C",     "84 °C",     "+0.12 t/ha"),
        ]
        for label, current, optimal, gain in suggestions:
            st.markdown(f"""
            <div style="
              display:flex; justify-content:space-between; align-items:center;
              padding:0.55rem 0; border-bottom:1px solid {BORDER};
              font-size:0.85rem;">
              <span style="color:{TEXT_PRIMARY};">{label}</span>
              <span style="color:{TEXT_MUTED};">{current} → {optimal}</span>
              <span style="color:{JD_GREEN}; font-weight:600;">{gain}</span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SEMÁFORO
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    _, center, _ = st.columns([1, 2, 1])

    with center:
        status_display = {
            "green":  (JD_GREEN,  "PROCEDER",           "Condiciones favorables para iniciar la cosecha."),
            "yellow": (JD_YELLOW, "PRECAUCIÓN",          "Condiciones dentro del rango normal. Monitorear."),
            "red":    ("#B94040", "REVISAR ANTES DE INICIAR", "Condiciones fuera del rango esperado."),
        }
        indicator_color, status_word, status_desc = status_display[REC_LEVEL]

        st.markdown(f"""
        <div style="text-align:center; padding:3rem 0 2rem 0;">
          <div style="
            width:140px; height:140px; border-radius:50%;
            background:{indicator_color}1a; border:3px solid {indicator_color};
            display:flex; align-items:center; justify-content:center;
            margin:0 auto 1.5rem auto;">
            <div style="
              width:90px; height:90px; border-radius:50%;
              background:{indicator_color};"></div>
          </div>
          <div style="
            font-size:1.1rem; font-weight:700; color:{indicator_color};
            letter-spacing:2px; margin-bottom:0.6rem;">
            {status_word}
          </div>
          <div style="font-size:0.95rem; color:{TEXT_MUTED}; max-width:340px; margin:0 auto;">
            {status_desc}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:0.7rem; color:{TEXT_MUTED}; text-transform:uppercase; '
            f'letter-spacing:1px; margin-bottom:1rem; text-align:center;">Variables más alejadas del óptimo</p>',
            unsafe_allow_html=True,
        )

        review_vars = [
            ("Humedad del grano", "18%",       "14%"),
            ("Fertilizante",      "120 kg/ha",  "200 kg/ha"),
            ("Temp. del motor",   "85 °C",       "84 °C"),
        ]
        chip_cols = st.columns(3)
        for i, (label, curr, opt) in enumerate(review_vars):
            with chip_cols[i]:
                st.markdown(f"""
                <div style="
                  background:{SURFACE}; border:1px solid {BORDER};
                  border-radius:6px; padding:1rem; text-align:center;">
                  <div style="color:{TEXT_MUTED}; font-size:0.72rem; margin-bottom:0.4rem; text-transform:uppercase; letter-spacing:0.5px;">{label}</div>
                  <div style="color:{TEXT_PRIMARY}; font-weight:600; font-size:1.15rem;">{curr}</div>
                  <div style="color:{TEXT_MUTED}; font-size:0.78rem; margin-top:0.3rem;">óptimo: {opt}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align:center; margin-top:2rem; color:{TEXT_MUTED}; font-size:0.82rem;">
          Confianza del modelo: <span style="color:{TEXT_PRIMARY}; font-weight:600;">{PRED_CONFIANZA}%</span>
          &nbsp;·&nbsp; 20 cosechas de referencia
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — HISTORIAL DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        f'<p style="font-size:0.7rem; color:{TEXT_MUTED}; text-transform:uppercase; '
        f'letter-spacing:1px; margin-bottom:1.2rem;">Rendimiento del modelo — validación leave-one-out en 30 cosechas</p>',
        unsafe_allow_html=True,
    )

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Error absoluto medio", f"{MAE:.2f} t/ha")
    mc2.metric("Cosechas validadas", "30")
    mc3.metric("Historial total", "200")

    st.markdown("<hr>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(
            x=VAL_DATA["Real"],
            y=VAL_DATA["Predicho"],
            mode="markers",
            marker=dict(size=8, color=JD_GREEN, opacity=0.8, line=dict(width=0)),
            name="Cosecha",
            hovertemplate="Real: %{x:.2f}<br>Predicho: %{y:.2f}<extra></extra>",
        ))
        fig_val.add_trace(go.Scatter(
            x=[4.5, 8.5], y=[4.5, 8.5],
            mode="lines",
            line=dict(color=BORDER, width=1.5, dash="dot"),
            name="Referencia perfecta",
        ))
        fig_val.update_layout(
            title=dict(text="Predicho vs Real", font=dict(size=13, color=TEXT_MUTED)),
            xaxis_title="Rendimiento real (t/ha)",
            yaxis_title="Rendimiento predicho (t/ha)",
            template="plotly_dark",
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            margin=dict(l=0, r=0, t=36, b=0),
            legend=dict(font=dict(color=TEXT_MUTED, size=11)),
            xaxis=dict(gridcolor=BORDER),
            yaxis=dict(gridcolor=BORDER),
        )
        st.plotly_chart(fig_val, use_container_width=True)

    with chart_col2:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=HIST_RENDIMIENTOS["rendimiento_real"],
            nbinsx=28,
            marker_color=JD_GREEN,
            opacity=0.7,
            name="Historial",
        ))
        fig_hist.add_vline(
            x=PRED_RENDIMIENTO,
            line_color=JD_YELLOW,
            line_width=2,
            line_dash="dash",
            annotation_text=f"{PRED_RENDIMIENTO:.2f} t/ha",
            annotation_font_color=JD_YELLOW,
            annotation_font_size=11,
        )
        fig_hist.update_layout(
            title=dict(text="Distribución histórica de rendimiento", font=dict(size=13, color=TEXT_MUTED)),
            xaxis_title="Rendimiento (t/ha)",
            yaxis_title="Frecuencia",
            template="plotly_dark",
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            margin=dict(l=0, r=0, t=36, b=0),
            showlegend=False,
            xaxis=dict(gridcolor=BORDER),
            yaxis=dict(gridcolor=BORDER),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(f"""
    <div style="
      background:{SURFACE}; border:1px solid {BORDER};
      border-radius:6px; padding:1.2rem 1.4rem; margin-top:0.5rem;
      font-size:0.88rem; color:{TEXT_MUTED}; line-height:1.7;">
      El modelo usa las 20 cosechas históricas más similares al escenario actual
      (distancia euclidiana normalizada). No hay algoritmos de ML externos —
      cada predicción es completamente trazable a registros reales.
      Error promedio en validación: <span style="color:{TEXT_PRIMARY}; font-weight:600;">{MAE:.2f} t/ha</span>.
    </div>
    """, unsafe_allow_html=True)
