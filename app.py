import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─── Brand palette ────────────────────────────────────────────────────────────
JD_GREEN       = "#367C2B"
JD_YELLOW      = "#FFDE00"
DARK_BG        = "#1C1C1C"
CARD_BG        = "#2A2A2A"
TEXT_PRIMARY   = "#FFFFFF"
TEXT_SECONDARY = "#AAAAAA"

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroIntel — John Deere",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  .stApp {{ background-color: {DARK_BG}; }}
  .main .block-container {{ padding-top: 1.2rem; max-width: 1400px; }}

  h1, h2, h3 {{ color: {TEXT_PRIMARY}; }}
  p, label, div {{ color: {TEXT_SECONDARY}; }}

  /* Metric cards */
  [data-testid="metric-container"] {{
    background-color: {CARD_BG};
    border-left: 4px solid {JD_GREEN};
    padding: 1rem 1.2rem;
    border-radius: 8px;
  }}
  [data-testid="stMetricValue"] {{ color: {JD_YELLOW} !important; font-size: 1.9rem !important; }}
  [data-testid="stMetricLabel"] {{ color: {TEXT_SECONDARY} !important; font-size: 0.8rem !important; }}
  [data-testid="stMetricDelta"] {{ font-size: 0.95rem !important; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background-color: {CARD_BG};
    border-radius: 10px;
    padding: 4px 6px;
    gap: 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
    color: {TEXT_SECONDARY};
    font-weight: 600;
    border-radius: 7px;
    padding: 6px 18px;
  }}
  .stTabs [aria-selected="true"] {{
    color: {JD_YELLOW} !important;
    background-color: {DARK_BG} !important;
  }}

  [data-testid="stSlider"] {{ padding: 0.2rem 0 0.5rem 0; }}
  hr {{ border-color: {JD_GREEN}55; }}

  .rec-box {{
    border-left: 4px solid;
    padding: 1rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 1rem 0;
  }}
  .impact-chip {{
    background-color: {CARD_BG};
    color: {JD_GREEN};
    border: 1px solid {JD_GREEN};
    border-radius: 20px;
    padding: 6px 16px;
    margin: 5px 0;
    display: block;
    font-size: 0.88rem;
    line-height: 1.6;
  }}
  .section-label {{
    color: {JD_YELLOW};
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.6rem;
    border-bottom: 1px solid {JD_GREEN}44;
    padding-bottom: 4px;
  }}
</style>
""", unsafe_allow_html=True)

# ─── Hardcoded sample data ────────────────────────────────────────────────────
# 20 "similar historical harvests" shown in the scatter chart
SIMILAR_HARVESTS = pd.DataFrame({
    "humedad_grano":    [14.2, 15.8, 16.1, 13.9, 17.2, 15.0, 18.3, 14.7, 16.9, 15.5,
                         17.8, 13.5, 16.4, 14.0, 18.1, 15.3, 17.0, 16.7, 14.5, 15.9],
    "rendimiento_real": [7.2,  6.8,  6.5,  7.5,  6.1,  7.0,  5.9,  7.3,  6.3,  6.9,
                         6.0,  7.6,  6.6,  7.4,  5.8,  7.1,  6.2,  6.4,  7.2,  6.7],
    "distance":         [0.05, 0.09, 0.12, 0.06, 0.18, 0.10, 0.22, 0.08, 0.15, 0.11,
                         0.20, 0.07, 0.13, 0.06, 0.23, 0.10, 0.17, 0.14, 0.09, 0.12],
})

# 200 historical rendimientos for the distribution histogram
HIST_RENDIMIENTOS = pd.DataFrame({
    "rendimiento_real": [
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
    ]
})

# 30 validation points for the predicted-vs-actual chart
VAL_DATA = pd.DataFrame({
    "Real":     [7.2, 6.8, 5.5, 7.5, 6.1, 6.9, 5.8, 7.3, 6.3, 7.0,
                 5.9, 7.6, 6.6, 7.4, 5.7, 7.1, 6.2, 6.4, 7.2, 6.7,
                 5.4, 7.0, 6.5, 7.3, 5.8, 6.9, 7.1, 6.3, 5.6, 7.4],
    "Predicho": [7.0, 6.9, 5.7, 7.3, 6.3, 7.1, 6.0, 7.2, 6.1, 6.8,
                 6.1, 7.4, 6.8, 7.2, 5.9, 7.0, 6.4, 6.6, 7.0, 6.9,
                 5.6, 6.8, 6.7, 7.1, 6.0, 7.1, 6.9, 6.5, 5.8, 7.2],
})

# Hardcoded prediction outputs (will be replaced by model later)
PRED_RENDIMIENTO = 6.85     # ton/ha
PRED_COSTO       = 682.0    # $
PRED_CONFIANZA   = 78       # %
PRED_DELTA_PCT   = +14.2    # % vs base
REC_LEVEL        = "green"  # green | yellow | red
HARDCODED_MAE    = 0.31     # ton/ha

# ─── App header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding: 0.5rem 0 0.2rem 0;">
  <span style="color:{JD_YELLOW}; font-size:2.4rem; font-weight:900;
               letter-spacing:3px; font-family:monospace;">
    🚜&nbsp; AGRO_INTEL
  </span><br>
  <span style="color:{TEXT_PRIMARY}; font-size:1.25rem; font-weight:600; letter-spacing:0.5px;">
    Simulador de Decisiones de Cosecha
  </span><br>
  <span style="color:{TEXT_SECONDARY}; font-size:0.82rem; letter-spacing:0.3px;">
    Reto 03 · Datos · John Deere × Tec de Monterrey
  </span>
</div>
<hr style="border:none; border-top:2px solid {JD_GREEN}; margin:0.6rem 0 1.2rem 0;">
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🌾  Simulador de Escenarios",
    "🚦  Semáforo Operacional",
    "📊  Historial del Modelo",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — SIMULADOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_in, col_out = st.columns([1, 1.7], gap="large")

    with col_in:
        st.markdown(f'<div class="section-label">Configura tu cosecha</div>', unsafe_allow_html=True)

        st.markdown(f"<span style='color:{TEXT_PRIMARY}; font-weight:600;'>Condiciones del grano</span>", unsafe_allow_html=True)
        st.slider("Humedad del grano (%)", 10, 30, 18)
        st.slider("Rendimiento esperado base (ton/ha)", 2.0, 12.0, 6.0, 0.1,
                  help="¿Cuánto esperas cosechar sin ajustes?")

        st.markdown(f"<span style='color:{TEXT_PRIMARY}; font-weight:600;'>Insumos</span>", unsafe_allow_html=True)
        st.slider("Fertilizante aplicado (kg/ha)", 0, 300, 120)
        st.slider("Pesticida aplicado (L/ha)", 0.0, 10.0, 2.5, 0.1)

        st.markdown(f"<span style='color:{TEXT_PRIMARY}; font-weight:600;'>Condiciones ambientales</span>", unsafe_allow_html=True)
        st.slider("Temperatura ambiente (°C)", 5, 45, 28)
        st.slider("Humedad del aire (%)", 20, 95, 60)

        st.markdown(f"<span style='color:{TEXT_PRIMARY}; font-weight:600;'>Estado de la máquina</span>", unsafe_allow_html=True)
        st.slider("Horas de motor acumuladas", 0, 5000, 1200)
        st.slider("Temperatura del motor (°C)", 60, 120, 85)
        st.slider("Velocidad de cosecha (km/h)", 2.0, 8.0, 5.0, 0.1)

    with col_out:
        st.markdown(
            f'<div class="section-label">Predicción basada en 20 cosechas similares</div>',
            unsafe_allow_html=True,
        )

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Rendimiento proyectado (ton/ha)",
            f"{PRED_RENDIMIENTO:.2f}",
            delta=f"{PRED_DELTA_PCT:+.1f}% vs base",
        )
        m2.metric("Costo operacional estimado ($)", f"${PRED_COSTO:,.0f}")
        m3.metric(
            "Confianza del modelo (%)",
            f"{PRED_CONFIANZA}%",
            help="Qué tan parecidas son las cosechas históricas usadas para esta predicción",
        )
        m4.metric("Cosechas similares", "20")

        # Scatter: similar harvests
        fig_scatter = px.scatter(
            SIMILAR_HARVESTS,
            x="humedad_grano",
            y="rendimiento_real",
            color="distance",
            color_continuous_scale=["#367C2B", "#FFDE00", "#CC4444"],
            labels={
                "humedad_grano":    "Humedad del grano (%)",
                "rendimiento_real": "Rendimiento real (ton/ha)",
                "distance":         "Distancia",
            },
            title="Las 20 cosechas históricas más similares a tu caso",
            template="plotly_dark",
            opacity=0.85,
        )
        # Current prediction star
        fig_scatter.add_trace(go.Scatter(
            x=[18],
            y=[PRED_RENDIMIENTO],
            mode="markers",
            marker=dict(symbol="star", size=20, color=JD_YELLOW,
                        line=dict(color="white", width=1.5)),
            name="Tu predicción",
        ))
        fig_scatter.update_layout(
            paper_bgcolor=CARD_BG,
            plot_bgcolor=CARD_BG,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(font=dict(color=TEXT_SECONDARY)),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Recommendation box
        _styles = {
            "green":  (JD_GREEN,  "#162b12", "🟢", "COSECHA FAVORABLE",
                       f"El modelo proyecta {PRED_RENDIMIENTO:.1f} ton/ha, "
                       f"un {abs(PRED_DELTA_PCT):.1f}% por encima de tu expectativa base. "
                       "Condiciones actuales son adecuadas."),
            "yellow": (JD_YELLOW, "#38320a", "🟡", "COSECHA EN RANGO ESPERADO",
                       f"Proyección de {PRED_RENDIMIENTO:.1f} ton/ha. "
                       "Dentro del rango normal. Revisa las variables en amarillo."),
            "red":    ("#CC3333", "#3d0c0c", "🔴", "COSECHA POR DEBAJO DE LO ESPERADO",
                       f"El modelo proyecta {PRED_RENDIMIENTO:.1f} ton/ha. "
                       "Considera ajustar las variables marcadas antes de comenzar."),
        }
        border_c, bg_c, icon, title_t, body_t = _styles[REC_LEVEL]
        st.markdown(f"""
        <div class="rec-box" style="border-color:{border_c}; background-color:{bg_c};">
          <div style="font-size:1.05rem; font-weight:700; color:{border_c};">
            {icon} &nbsp; {title_t}
          </div>
          <div style="color:{TEXT_PRIMARY}; margin-top:0.45rem; font-size:0.92rem; line-height:1.55;">
            {body_t}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Impact chips (hardcoded top-3)
        st.markdown(
            f'<div class="section-label" style="margin-top:1.2rem;">Ajustes con mayor impacto potencial</div>',
            unsafe_allow_html=True,
        )
        impact_chips = [
            ("reduces humedad a", "14%",      "+0.48 ton/ha"),
            ("aumentas fertilizante a", "200 kg/ha", "+0.31 ton/ha"),
            ("mantienes motor a", "84 °C",    "+0.12 ton/ha"),
        ]
        for verb, value, gain in impact_chips:
            st.markdown(
                f'<div class="impact-chip">↑ &nbsp; Si {verb} <b>{value}</b>, '
                f'ganarías <b>{gain}</b></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SEMÁFORO OPERACIONAL
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    _circle_cfg = {
        "green":  (JD_GREEN,  "0 0 50px #367C2B99", "✅", "Proceder. Condiciones favorables."),
        "yellow": (JD_YELLOW, "0 0 50px #FFDE0099", "⚠️", "Proceder con precaución. Monitorear humedad."),
        "red":    ("#CC3333", "0 0 50px #CC333399", "🛑", "Revisar condiciones antes de comenzar."),
    }
    circle_color, shadow, circle_icon, instruction = _circle_cfg[REC_LEVEL]

    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        st.markdown(f"""
        <div style="text-align:center; padding: 2rem 0 1rem 0;">
          <div style="
            width:200px; height:200px; border-radius:50%;
            background:{circle_color}; box-shadow:{shadow};
            display:flex; align-items:center; justify-content:center;
            font-size:80px; margin:0 auto 1.6rem auto;">
            {circle_icon}
          </div>
          <div style="
            font-size:1.55rem; font-weight:800;
            color:{circle_color}; line-height:1.35; margin-bottom:1.8rem;">
            {instruction}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Top 3 variables to review (hardcoded)
        st.markdown(
            f'<div class="section-label" style="text-align:center; margin-bottom:1rem;">Variables a revisar</div>',
            unsafe_allow_html=True,
        )
        review_vars = [
            ("Humedad del grano (%)", "18%",      "14%"),
            ("Fertilizante (kg/ha)",  "120 kg/ha", "200 kg/ha"),
            ("Temperatura del motor", "85 °C",     "84 °C"),
        ]
        chip_cols = st.columns(3)
        for i, (label, curr, opt) in enumerate(review_vars):
            with chip_cols[i]:
                st.markdown(f"""
                <div style="
                  background:{CARD_BG}; border:1px solid {circle_color};
                  border-radius:12px; padding:0.9rem 0.7rem; text-align:center;">
                  <div style="color:{TEXT_SECONDARY}; font-size:0.75rem; line-height:1.4;">{label}</div>
                  <div style="color:{TEXT_PRIMARY}; font-weight:700; font-size:1.2rem; margin:0.3rem 0;">{curr}</div>
                  <div style="color:{TEXT_SECONDARY}; font-size:0.78rem;">→ óptimo: {opt}</div>
                </div>
                """, unsafe_allow_html=True)

        # Confidence badge
        st.markdown(f"""
        <div style="text-align:center; margin-top:2rem;">
          <div style="
            background:{CARD_BG}; border-radius:20px;
            padding:0.55rem 1.4rem; display:inline-block;
            color:{TEXT_SECONDARY}; font-size:0.88rem;">
            Modelo:&nbsp;
            <b style="color:{JD_YELLOW};">{PRED_CONFIANZA}%</b>
            &nbsp;confianza&nbsp;·&nbsp;basado en 20 cosechas similares
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — HISTORIAL DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        f'<div style="font-size:1.45rem; font-weight:700; color:{TEXT_PRIMARY}; margin-bottom:0.3rem;">'
        "¿Por qué puedes confiar en esta predicción?"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{TEXT_SECONDARY}; margin-bottom:1.2rem; font-size:0.9rem;">'
        "Validación leave-one-out sobre 30 cosechas seleccionadas al azar del historial."
        "</div>",
        unsafe_allow_html=True,
    )

    # Summary metrics
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Error Absoluto Medio (MAE)", f"{HARDCODED_MAE:.2f} ton/ha")
    mc2.metric("Cosechas validadas", "30")
    mc3.metric("Historial total", "200 cosechas")

    st.markdown(f"""
    <div style="
      background:{CARD_BG}; border-left:4px solid {JD_GREEN};
      padding:0.75rem 1.2rem; border-radius:0 8px 8px 0;
      margin:0.6rem 0 1.4rem 0; color:{TEXT_SECONDARY}; font-size:0.92rem;">
      📏&nbsp; En promedio, el modelo se equivoca por
      <b style="color:{JD_YELLOW};">{HARDCODED_MAE:.2f} ton/ha</b>
    </div>
    """, unsafe_allow_html=True)

    # Predicted vs actual scatter
    fig_val = px.scatter(
        VAL_DATA,
        x="Real",
        y="Predicho",
        title="Predicho vs Real — 30 cosechas de validación",
        template="plotly_dark",
        color_discrete_sequence=[JD_GREEN],
        labels={
            "Real":     "Rendimiento Real (ton/ha)",
            "Predicho": "Rendimiento Predicho (ton/ha)",
        },
        opacity=0.85,
    )
    fig_val.add_trace(go.Scatter(
        x=[4.5, 8.5], y=[4.5, 8.5],
        mode="lines",
        name="Predicción perfecta",
        line=dict(color=JD_YELLOW, dash="dash", width=1.8),
    ))
    fig_val.update_layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(font=dict(color=TEXT_SECONDARY)),
    )
    st.plotly_chart(fig_val, use_container_width=True)

    # Rendimiento distribution histogram
    fig_hist = px.histogram(
        HIST_RENDIMIENTOS,
        x="rendimiento_real",
        nbins=30,
        title="Tu predicción en contexto: dónde cae respecto a todas las cosechas históricas",
        template="plotly_dark",
        color_discrete_sequence=[JD_GREEN],
        labels={
            "rendimiento_real": "Rendimiento (ton/ha)",
            "count":            "Frecuencia",
        },
    )
    fig_hist.add_vline(
        x=PRED_RENDIMIENTO,
        line_color=JD_YELLOW,
        line_width=2.5,
        line_dash="dash",
        annotation_text=f"Tu predicción: {PRED_RENDIMIENTO:.2f}",
        annotation_font_color=JD_YELLOW,
        annotation_position="top right",
    )
    fig_hist.update_layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Callout box
    st.markdown(f"""
    <div style="
      background:{CARD_BG}; border:1px solid {JD_GREEN}66;
      border-radius:10px; padding:1.3rem 1.6rem; margin-top:0.5rem;">
      <div style="color:{JD_YELLOW}; font-weight:700; font-size:1rem; margin-bottom:0.5rem;">
        📊 &nbsp;Sobre este modelo
      </div>
      <div style="color:{TEXT_SECONDARY}; font-size:0.92rem; line-height:1.65;">
        Este modelo fue construido con
        <b style="color:{TEXT_PRIMARY};">200 cosechas históricas</b>.
        La predicción actual se basa en las
        <b style="color:{TEXT_PRIMARY};">20 más similares</b>
        a tus condiciones, usando distancia euclidiana normalizada.
        Error promedio histórico:
        <b style="color:{JD_YELLOW};">{HARDCODED_MAE:.2f} ton/ha</b>.
        <br><br>
        Ningún algoritmo de ML externo es utilizado —
        la predicción es completamente interpretable y trazable
        a cosechas reales pasadas.
      </div>
    </div>
    """, unsafe_allow_html=True)
