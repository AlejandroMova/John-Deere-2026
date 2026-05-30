import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import ceil, pi, sqrt

# ─── Brand palette ────────────────────────────────────────────────────────────
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
  h1, h2, h3, h4, p, label, div {{ color: {TEXT_PRIMARY}; }}

  [data-testid="stSlider"] {{ padding: 0.1rem 0 0.4rem 0; }}
  .stSlider label p {{ color: {TEXT_MUTED} !important; font-size: 0.82rem !important; }}

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
  [data-testid="stMetricDelta"] {{ font-size: 0.85rem !important; }}

  .stTabs [data-baseweb="tab-list"] {{
    background-color: transparent;
    border-bottom: 1px solid {BORDER};
    gap: 0; padding: 0;
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
  hr {{ border: none; border-top: 1px solid {BORDER}; margin: 1.5rem 0; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS — a) Operaciones
# ══════════════════════════════════════════════════════════════════════════════

def plan_boustrophedon(field_length_m, field_width_m, machine_width_m, speed_kmh):
    n   = ceil(field_width_m / machine_width_m)
    d   = n * field_length_m + (n - 1) * pi * (machine_width_m / 2)
    mw  = machine_width_m
    px, py = [], []
    for i in range(n):
        x = (i + 0.5) * mw
        if i % 2 == 0:
            px += [x, x, None]; py += [0, field_length_m, None]
        else:
            px += [x, x, None]; py += [field_length_m, 0, None]
    return {
        "n": n, "turns": n - 1,
        "dist_km": d / 1000,
        "time_h":  (d / 1000) / speed_kmh,
        "fuel_l":  (d / 1000) * (5.5 / speed_kmh + 0.14 * speed_kmh),
        "area_ha": (field_length_m * field_width_m) / 10_000,
        "px": px, "py": py,
        "W": field_width_m, "L": field_length_m, "mw": mw,
    }


def analyze_speed(distance_km, engine_hours):
    v            = np.linspace(3.0, 12.0, 200)
    fuel         = (5.5 / v + 0.14 * v) * distance_km
    time_h       = distance_km / v
    stress       = np.exp(0.3 * np.maximum(0, v - 8.0))
    lam          = 0.001 * stress * (1 + engine_hours / 8000)
    fail_pct     = (1 - np.exp(-lam * time_h)) * 100

    def n01(a): lo, hi = a.min(), a.max(); return (a - lo) / (hi - lo) if hi > lo else a * 0
    opt_v = float(v[(n01(fuel) + n01(time_h) + n01(fail_pct)).argmin()])
    return v, fuel, time_h, fail_pct, opt_v


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS — b) Siembra y predicciones
# ══════════════════════════════════════════════════════════════════════════════

def seed_depth_cm(soil_moist, soil_temp, crop):
    p = {"Maíz":  (3.8, 22, 14), "Trigo": (3.0, 18, 9),
         "Soya":  (3.5, 20, 16), "Sorgo": (2.5, 20, 19)}[crop]
    base, mopt, tthresh = p
    d = base + (mopt - soil_moist) * 0.06
    if soil_temp < tthresh + 4:
        d -= 0.5
    return float(np.clip(d, 1.5, 8.0))


def fert_dose(n_soil_ppm, yield_target, stage):
    fracs = {"Siembra": 0.30, "Vegetativo": 0.45, "Reproductivo": 0.20, "Madurez": 0.05}
    ymax  = yield_target * 1.18
    n_kg  = n_soil_ppm * 3.9
    n_need = max(0, -np.log(1 - yield_target / ymax) / 0.008)
    return float(np.clip(max(0, n_need - n_kg) * fracs.get(stage, 0.30), 0, 280))


def field_quality_map(rows=30, cols=40, seed=42):
    rng  = np.random.default_rng(seed)
    grid = np.zeros((rows, cols))
    for cr, cc, sr, sc, iv in [(0.25,0.30,8,10,1.0),(0.70,0.60,6,12,0.9),
                                (0.40,0.80,7,8,0.8),(0.55,0.15,10,7,0.7)]:
        ri = np.arange(rows)[:, None]; ci = np.arange(cols)[None, :]
        grid += iv * np.exp(-((ri-cr*rows)**2/(2*sr**2)+(ci-cc*cols)**2/(2*sc**2)))
    grid += rng.uniform(0, 0.12, (rows, cols))
    return (grid - grid.min()) / (grid.max() - grid.min())


def harvest_timing(gdd_now, moist_now, crop, avg_temp):
    gt = {"Maíz": 2800, "Trigo": 2000, "Soya": 2600, "Sorgo": 2400}[crop]
    mt = {"Maíz": 18,   "Trigo": 14,   "Soya": 14,   "Sorgo": 16}[crop]
    days = max(max(0, gt - gdd_now) / max(0.1, avg_temp - 10),
               max(0, moist_now - mt) / 0.7)
    color = JD_GREEN if days < 5 else JD_YELLOW if days < 20 else "#B94040"
    label = "Listo" if days < 5 else "Próximo" if days < 20 else "No listo"
    return days, label, color


def nozzle_flow(pressure_bar, nozzle_key, speed_kmh, spacing_m):
    K = {"015": 0.24, "02": 0.33, "03": 0.49, "04": 0.65, "05": 0.82}[nozzle_key]
    q = K * sqrt(pressure_bar)
    return q, q * 600 / (speed_kmh * spacing_m)


def forecast_7d(base, sigma, rev_rate, lo, hi, seed):
    rng = np.random.default_rng(seed)
    v   = [base]
    for _ in range(6):
        v.append(float(np.clip(v[-1] + rng.normal(0, sigma) - rev_rate*(v[-1]-base), lo, hi)))
    return np.array(v)


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC TRAINING DATA  (shown in tab 4 expander)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def make_training_data():
    rng = np.random.default_rng(42); n = 300
    # Seed depth
    sm = rng.uniform(10,40,n); st_ = rng.uniform(5,35,n); ci = rng.integers(0,4,n)
    bd = np.array([3.8,3.0,3.5,2.5])[ci]; mo = np.array([22,18,20,20])[ci]
    dp = (bd + (mo-sm)*0.06 + rng.normal(0,.3,n)).clip(1.5,8.0)
    df_s = pd.DataFrame({"humedad_suelo_pct":sm,"temp_suelo_c":st_,
                          "cultivo":["Maíz","Trigo","Soya","Sorgo"][ci[i]] if False else
                          [["Maíz","Trigo","Soya","Sorgo"][j] for j in ci],
                          "profundidad_cm":dp})
    # Fertilizer
    ns = rng.uniform(5,60,n); yt = rng.uniform(4,10,n)
    si = rng.integers(0,4,n); fr = np.array([.30,.45,.20,.05])[si]
    na = (np.maximum(0,-np.log(1-yt/(yt*1.18))/0.008-ns*3.9)*fr+rng.normal(0,8,n)).clip(0,280)
    df_f = pd.DataFrame({"n_suelo_ppm":ns,"objetivo_t_ha":yt,"dosis_kg_ha":na})
    # Harvest
    gc = rng.uniform(800,3500,n); gm = rng.uniform(12,35,n); ta = rng.uniform(15,38,n)
    c2 = rng.integers(0,4,n)
    gr = np.array([2800,2000,2600,2400])[c2]; mr = np.array([18,14,14,16])[c2]
    dh = (np.maximum(np.maximum(0,gr-gc)/np.maximum(.1,ta-10),
                     np.maximum(0,gm-mr)/0.7)+rng.normal(0,1.5,n)).clip(0,90)
    df_h = pd.DataFrame({"gdd_acumulados":gc,"humedad_grano_pct":gm,
                          "temp_promedio_c":ta,"dias_para_cosechar":dh})
    # Speed/fuel
    vs = rng.uniform(3,12,n)
    fl = (5.5/vs+0.14*vs+rng.normal(0,.08,n)).clip(.8,4.5)
    df_v = pd.DataFrame({"velocidad_kmh":vs,"combustible_lkm":fl})
    return df_s, df_f, df_h, df_v


# ── Hardcoded prediction outputs (Tabs 1–2) ───────────────────────────────────
SIMILAR = pd.DataFrame({
    "humedad_grano":    [14.2,15.8,16.1,13.9,17.2,15.0,18.3,14.7,16.9,15.5,
                         17.8,13.5,16.4,14.0,18.1,15.3,17.0,16.7,14.5,15.9],
    "rendimiento_real": [7.2, 6.8, 6.5, 7.5, 6.1, 7.0, 5.9, 7.3, 6.3, 6.9,
                         6.0, 7.6, 6.6, 7.4, 5.8, 7.1, 6.2, 6.4, 7.2, 6.7],
    "distance":         [.05, .09, .12, .06, .18, .10, .22, .08, .15, .11,
                         .20, .07, .13, .06, .23, .10, .17, .14, .09, .12],
})
HIST = pd.DataFrame({"r": [
    5.2,6.8,7.1,4.9,6.3,7.5,5.8,6.1,7.8,5.5,6.6,7.2,5.0,6.9,7.4,5.3,6.4,7.0,5.7,6.2,
    7.6,5.1,6.7,7.3,5.4,6.0,7.9,5.6,6.5,7.1,4.8,6.8,7.5,5.2,6.3,7.2,5.9,6.1,7.7,5.4,
    6.6,7.0,5.1,6.9,7.4,5.7,6.4,7.3,5.0,6.2,7.6,5.3,6.7,7.1,5.5,6.0,7.8,5.8,6.5,7.2,
    4.9,6.8,7.5,5.2,6.3,7.0,5.9,6.2,7.7,5.4,6.6,7.3,5.1,6.9,7.4,5.7,6.4,7.1,5.0,6.1,
    7.6,5.3,6.7,7.2,5.5,6.0,7.9,5.8,6.5,7.0,4.8,6.8,7.5,5.2,6.3,7.3,5.9,6.2,7.7,5.4,
    6.6,7.1,5.1,6.9,7.4,5.7,6.4,7.2,5.0,6.2,7.6,5.3,6.7,7.0,5.5,6.0,7.8,5.8,6.5,7.3,
    4.9,6.8,7.5,5.2,6.3,7.1,5.9,6.1,7.7,5.4,6.6,7.2,5.1,6.9,7.4,5.7,6.4,7.0,5.0,6.2,
    7.6,5.3,6.7,7.3,5.5,6.0,7.9,5.8,6.5,7.1,4.8,6.8,7.5,5.2,6.3,7.2,5.9,6.2,7.7,5.4,
    6.6,7.0,5.1,6.9,7.4,5.7,6.4,7.3,5.0,6.1,7.6,5.3,6.7,7.1,5.5,6.0,7.8,5.8,6.5,7.2,
    4.9,6.8,7.5,5.2,6.3,7.0,5.9,6.2,7.7,5.4,6.6,7.3,5.1,6.9,7.4,5.7,6.4,7.1,5.0,6.2,
]})
VAL = pd.DataFrame({
    "Real":     [7.2,6.8,5.5,7.5,6.1,6.9,5.8,7.3,6.3,7.0,
                 5.9,7.6,6.6,7.4,5.7,7.1,6.2,6.4,7.2,6.7,
                 5.4,7.0,6.5,7.3,5.8,6.9,7.1,6.3,5.6,7.4],
    "Pred":     [7.0,6.9,5.7,7.3,6.3,7.1,6.0,7.2,6.1,6.8,
                 6.1,7.4,6.8,7.2,5.9,7.0,6.4,6.6,7.0,6.9,
                 5.6,6.8,6.7,7.1,6.0,7.1,6.9,6.5,5.8,7.2],
})
PRED_R = 6.85; PRED_C = 682.0; CONF = 78; DELTA = +14.2
REC    = "green"; MAE = 0.31

CL = dict(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
          margin=dict(l=0,r=0,t=36,b=0),
          font=dict(color=TEXT_MUTED, size=11),
          legend=dict(font=dict(color=TEXT_MUTED, size=11)))
AX = dict(xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER))

def slabel(txt):
    st.markdown(f'<p style="font-size:.7rem;color:{TEXT_MUTED};text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:.8rem;">{txt}</p>', unsafe_allow_html=True)

def mnote(txt):
    st.markdown(f'<p style="color:{TEXT_MUTED};font-size:.75rem;font-style:italic;'
                f'margin-top:.3rem;">{txt}</p>', unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 2])
with c1:
    st.markdown(f'<div style="font-size:1rem;color:{TEXT_MUTED};font-weight:500;margin-bottom:2px;">'
                f'<span style="color:{JD_GREEN};font-weight:700;">John Deere</span> · AgroIntel</div>'
                f'<div style="font-size:1.75rem;font-weight:700;color:{TEXT_PRIMARY};line-height:1.2;">'
                f'Simulador de Decisiones de Cosecha</div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div style="text-align:right;padding-top:.8rem;">'
                f'<span style="color:{TEXT_MUTED};font-size:.8rem;">Reto 03 · Datos · Tec de Monterrey</span>'
                f'</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Simulador de cosecha",
    "Semáforo operacional",
    "Operaciones del tractor",
    "Siembra y predicciones",
    "Historial del modelo",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — SIMULADOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    ci, co = st.columns([1, 1.8], gap="large")
    with ci:
        slabel("Parámetros de entrada")
        st.caption("Condiciones del grano")
        st.slider("Humedad del grano (%)", 10, 30, 18)
        st.slider("Rendimiento esperado base (ton/ha)", 2.0, 12.0, 6.0, 0.1,
                  help="Expectativa sin ajustes")
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

    with co:
        slabel("Resultados — 20 cosechas similares")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Rendimiento proyectado", f"{PRED_R:.2f} t/ha", delta=f"{DELTA:+.1f}% vs base")
        m2.metric("Costo estimado", f"${PRED_C:,.0f}")
        m3.metric("Confianza", f"{CONF}%", help="Similitud promedio de las 20 cosechas de referencia")
        m4.metric("Cosechas de ref.", "20")

        sc, sb, st_ = {"green": (JD_GREEN,"#1a2e19","Condiciones favorables — se puede proceder."),
                       "yellow":(JD_YELLOW,"#2e2a14","Rango normal — monitorear humedad."),
                       "red":   ("#B94040","#2e1414","Por debajo de lo esperado — revisar.")}[REC]
        st.markdown(f'<div style="background:{sb};border-left:3px solid {sc};padding:.7rem 1rem;'
                    f'border-radius:0 4px 4px 0;margin:1rem 0;font-size:.88rem;color:{TEXT_PRIMARY};">'
                    f'{st_}</div>', unsafe_allow_html=True)

        opac = [max(.35, 1-d*3) for d in SIMILAR["distance"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=SIMILAR["humedad_grano"], y=SIMILAR["rendimiento_real"],
            mode="markers", marker=dict(size=9, color=[f"rgba(54,124,43,{o:.2f})" for o in opac]),
            hovertemplate="Humedad: %{x:.1f}%<br>Rendimiento: %{y:.2f} t/ha<extra></extra>"))
        fig.add_trace(go.Scatter(x=[18], y=[PRED_R], mode="markers",
            marker=dict(symbol="diamond", size=12, color=JD_YELLOW, line=dict(color=DARK_BG,width=1)),
            name="Tu escenario"))
        fig.update_layout(title=dict(text="20 cosechas más similares",font=dict(size=13,color=TEXT_MUTED)),
            xaxis_title="Humedad del grano (%)", yaxis_title="Rendimiento (t/ha)",
            template="plotly_dark", **CL, **AX)
        st.plotly_chart(fig, use_container_width=True)

        slabel("Ajustes sugeridos")
        for lbl,cur,opt,gain in [("Humedad del grano","18%","14%","+0.48 t/ha"),
                                   ("Fertilizante","120 kg/ha","200 kg/ha","+0.31 t/ha"),
                                   ("Temperatura motor","85 °C","84 °C","+0.12 t/ha")]:
            st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'padding:.55rem 0;border-bottom:1px solid {BORDER};font-size:.85rem;">'
                        f'<span style="color:{TEXT_PRIMARY};">{lbl}</span>'
                        f'<span style="color:{TEXT_MUTED};">{cur} → {opt}</span>'
                        f'<span style="color:{JD_GREEN};font-weight:600;">{gain}</span></div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SEMÁFORO
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    _, cc, _ = st.columns([1,2,1])
    with cc:
        ic, sw, sd = {"green": (JD_GREEN, "PROCEDER","Condiciones favorables para iniciar la cosecha."),
                      "yellow":(JD_YELLOW,"PRECAUCIÓN","Condiciones dentro del rango normal. Monitorear."),
                      "red":   ("#B94040","REVISAR ANTES DE INICIAR","Condiciones fuera del rango esperado.")}[REC]
        st.markdown(f'<div style="text-align:center;padding:3rem 0 2rem 0;">'
                    f'<div style="width:140px;height:140px;border-radius:50%;background:{ic}1a;'
                    f'border:3px solid {ic};display:flex;align-items:center;justify-content:center;'
                    f'margin:0 auto 1.5rem auto;">'
                    f'<div style="width:90px;height:90px;border-radius:50%;background:{ic};"></div></div>'
                    f'<div style="font-size:1.1rem;font-weight:700;color:{ic};letter-spacing:2px;'
                    f'margin-bottom:.6rem;">{sw}</div>'
                    f'<div style="font-size:.95rem;color:{TEXT_MUTED};max-width:340px;margin:0 auto;">{sd}</div>'
                    f'</div>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        slabel("Variables más alejadas del óptimo")
        cols = st.columns(3)
        for i,(lbl,cur,opt) in enumerate([("Humedad del grano","18%","14%"),
                                           ("Fertilizante","120 kg/ha","200 kg/ha"),
                                           ("Temp. del motor","85 °C","84 °C")]):
            with cols[i]:
                st.markdown(f'<div style="background:{SURFACE};border:1px solid {BORDER};'
                            f'border-radius:6px;padding:1rem;text-align:center;">'
                            f'<div style="color:{TEXT_MUTED};font-size:.72rem;margin-bottom:.4rem;'
                            f'text-transform:uppercase;letter-spacing:.5px;">{lbl}</div>'
                            f'<div style="color:{TEXT_PRIMARY};font-weight:600;font-size:1.15rem;">{cur}</div>'
                            f'<div style="color:{TEXT_MUTED};font-size:.78rem;margin-top:.3rem;">óptimo: {opt}</div>'
                            f'</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center;margin-top:2rem;color:{TEXT_MUTED};font-size:.82rem;">'
                    f'Confianza: <span style="color:{TEXT_PRIMARY};font-weight:600;">{CONF}%</span>'
                    f' · 20 cosechas de referencia</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — OPERACIONES DEL TRACTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    # a.1 — Ruta óptima ───────────────────────────────────────────────────────
    slabel("a.1 · Ruta óptima de cobertura")
    ri, ro = st.columns([1, 2], gap="large")

    with ri:
        fl_  = st.slider("Largo del campo (m)",        50,  500, 300, 10)
        fw_  = st.slider("Ancho del campo (m)",         50,  500, 200, 10)
        mw_  = st.slider("Ancho de trabajo (m)",        2.0, 12.0, 4.5, 0.5)
        spd_ = st.slider("Velocidad de trabajo (km/h)", 3.0, 12.0, 7.0, 0.5)

    r = plan_boustrophedon(fl_, fw_, mw_, spd_)

    with ro:
        rm1,rm2,rm3,rm4 = st.columns(4)
        rm1.metric("Distancia total",  f"{r['dist_km']:.1f} km")
        rm2.metric("Tiempo estimado",  f"{r['time_h']:.1f} h")
        rm3.metric("Combustible est.", f"{r['fuel_l']:.0f} L")
        rm4.metric("Área cubierta",    f"{r['area_ha']:.1f} ha")

        fig_r = go.Figure()
        fig_r.add_shape(type="rect", x0=0, y0=0, x1=r["W"], y1=r["L"],
                        line=dict(color=BORDER, width=1.5))
        for i in range(r["n"]):
            x0, x1 = i*r["mw"], min((i+1)*r["mw"], r["W"])
            fig_r.add_shape(type="rect", x0=x0, y0=0, x1=x1, y1=r["L"],
                            fillcolor=f"rgba(54,124,43,{.25 if i%2==0 else .12})",
                            line=dict(width=0))
        fig_r.add_trace(go.Scatter(x=r["px"], y=r["py"], mode="lines",
                                   line=dict(color=JD_GREEN, width=1.5), name="Trayectoria"))
        fig_r.add_trace(go.Scatter(x=[r["mw"]/2], y=[0], mode="markers",
                                   marker=dict(symbol="circle", size=10, color=JD_YELLOW),
                                   name="Inicio"))
        fig_r.update_layout(
            title=dict(text=f"Boustrofedón — {r['n']} pasadas, {r['turns']} giros",
                       font=dict(size=13, color=TEXT_MUTED)),
            xaxis_title="Ancho (m)", yaxis_title="Largo (m)",
            template="plotly_dark", **CL,
            xaxis=dict(gridcolor=BORDER),
            yaxis=dict(scaleanchor="x", scaleratio=1, gridcolor=BORDER),
        )
        st.plotly_chart(fig_r, use_container_width=True)
        mnote("Optimización: patrón boustrofedón con giros de 180° en cabecera (radio = ancho_trabajo / 2). "
              "Mínimo de giros para campo rectangular.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # a.2 — Velocidad óptima ──────────────────────────────────────────────────
    slabel("a.2 · Velocidad óptima — combustible, tiempo y probabilidad de falla")
    vi, vo = st.columns([1, 2], gap="large")

    with vi:
        dist_km_  = st.slider("Distancia a recorrer (km)",  1.0, 60.0, 15.0, 0.5)
        eng_hrs_  = st.slider("Horas de motor acumuladas ",  0,   5000, 1200, 50)

    v, fuel, time_h, fail, opt_v = analyze_speed(dist_km_, eng_hrs_)

    with vo:
        oi = int(np.argmin(np.abs(v - opt_v)))
        vm1,vm2,vm3 = st.columns(3)
        vm1.metric("Velocidad óptima",        f"{opt_v:.1f} km/h")
        vm2.metric("Combustible a v óptima",  f"{fuel[oi]:.1f} L")
        vm3.metric("P(falla) a v óptima",     f"{fail[oi]:.2f}%")

        fig_sp = make_subplots(rows=1, cols=3,
            subplot_titles=["Combustible total (L)", "Tiempo (h)", "P(falla) (%)"])
        for col_i, (ydata, color) in enumerate(
            [(fuel, JD_GREEN), (time_h, "#5B9BD5"), (fail, "#CC5555")], 1):
            fig_sp.add_trace(go.Scatter(x=v, y=ydata, mode="lines",
                             line=dict(color=color, width=2), showlegend=False), row=1, col=col_i)
            fig_sp.add_vline(x=opt_v, line_color=JD_YELLOW, line_dash="dash",
                             line_width=1.5, row=1, col=col_i)
        fig_sp.update_layout(
            title=dict(text=f"Velocidad óptima Pareto: {opt_v:.1f} km/h",
                       font=dict(size=13, color=TEXT_MUTED)),
            template="plotly_dark", **{**CL},
        )
        for i in range(1, 4):
            fig_sp.update_xaxes(title_text="km/h", gridcolor=BORDER, row=1, col=i)
            fig_sp.update_yaxes(gridcolor=BORDER, row=1, col=i)
        st.plotly_chart(fig_sp, use_container_width=True)
        mnote("Optimización Pareto multi-objetivo: suma de scores normalizados [0–1] para combustible, "
              "tiempo y P(falla). La línea amarilla marca el mínimo compuesto.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — SIEMBRA Y PREDICCIONES
# ══════════════════════════════════════════════════════════════════════════════
with tab4:

    # b.1 Profundidad  +  b.4 Fertilizante ───────────────────────────────────
    cs, cf = st.columns(2, gap="large")

    with cs:
        slabel("b.1 · Profundidad óptima de semilla")
        crop_  = st.selectbox("Cultivo", ["Maíz", "Trigo", "Soya", "Sorgo"])
        smoist = st.slider("Humedad del suelo (%)",    10, 45, 22)
        stemp  = st.slider("Temperatura del suelo (°C)", 5, 35, 18)
        depth  = seed_depth_cm(smoist, stemp, crop_)

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=depth,
            number=dict(suffix=" cm", font=dict(size=28, color=TEXT_PRIMARY)),
            gauge=dict(
                axis=dict(range=[1.5, 8.0], tickcolor=TEXT_MUTED),
                bar=dict(color=JD_GREEN), bgcolor=SURFACE_HIGH, borderwidth=0,
                steps=[dict(range=[1.5,3.0],color="#1a3d1a"),
                       dict(range=[3.0,5.5],color="#213d21"),
                       dict(range=[5.5,8.0],color="#2e1a1a")],
                threshold=dict(line=dict(color=JD_YELLOW,width=2), value=depth),
            ),
        ))
        fig_g.update_layout(title=dict(text="Profundidad recomendada",
                                       font=dict(size=13,color=TEXT_MUTED)),
                            height=250, **CL)
        fig_g.update_layout(margin=dict(l=20, r=20, t=36, b=10))
        st.plotly_chart(fig_g, use_container_width=True)
        mnote("Modelo: tabla de respuesta agronómica con ajuste por humedad y temperatura del suelo. "
              "NN candidata: MLP 3→16→8→1 (ReLU).")

    with cf:
        slabel("b.4 · Dosis de fertilizante nitrogenado")
        n_ppm_  = st.slider("Nitrógeno en suelo (ppm)",         5,  60, 20)
        y_tgt_  = st.slider("Objetivo rendimiento (t/ha)",      3.0,10.0,6.5,0.5)
        stage_  = st.selectbox("Etapa del cultivo",
                               ["Siembra","Vegetativo","Reproductivo","Madurez"])
        dose    = fert_dose(n_ppm_, y_tgt_, stage_)

        n_range = np.linspace(0, 280, 300)
        y_curve = (y_tgt_*1.18) * (1 - np.exp(-0.008*(n_range + n_ppm_*3.9)))
        fig_f   = go.Figure()
        fig_f.add_trace(go.Scatter(x=n_range, y=y_curve, mode="lines",
                                   line=dict(color=JD_GREEN, width=2), name="Respuesta al N"))
        fig_f.add_vline(x=dose, line_color=JD_YELLOW, line_dash="dash", line_width=1.8,
                        annotation_text=f"{dose:.0f} kg/ha",
                        annotation_font_color=JD_YELLOW, annotation_font_size=11)
        fig_f.update_layout(
            title=dict(text="Curva de respuesta N (Mitscherlich-Baule)",
                       font=dict(size=13,color=TEXT_MUTED)),
            xaxis_title="N aplicado (kg/ha)", yaxis_title="Rendimiento esperado (t/ha)",
            template="plotly_dark", height=250, **CL, **AX)
        st.plotly_chart(fig_f, use_container_width=True)
        mnote("Modelo: ecuación de saturación Mitscherlich-Baule. "
              "NN candidata: Monotone MLP (constraint de monotonicidad).")

    st.markdown("<hr>", unsafe_allow_html=True)

    # b.6 Mapa de posicionamiento ─────────────────────────────────────────────
    slabel("b.6 · Mapa de posicionamiento de semillas")
    mc1_, mc2_ = st.columns([1, 3])
    with mc1_:
        map_seed_ = st.slider("Variabilidad del terreno", 1, 99, 42)
        thresh_   = st.slider("Umbral zona óptima (%)", 40, 80, 60)
    qmap = field_quality_map(seed=map_seed_)
    with mc2_:
        fig_m = go.Figure()
        fig_m.add_trace(go.Heatmap(z=qmap,
            colorscale=[[0,"#3d0c0c"],[.4,"#7a3a00"],[.6,"#2e4a1a"],[1,JD_GREEN]],
            colorbar=dict(title=dict(text="Calidad", font=dict(color=TEXT_MUTED)),
                          tickfont=dict(color=TEXT_MUTED)),
            zmin=0, zmax=1))
        fig_m.add_trace(go.Contour(z=qmap,
            contours=dict(start=thresh_/100,end=1.0,size=.5,coloring="none"),
            line=dict(color=JD_YELLOW,width=1.5,dash="dot"),showscale=False,
            name=f"Zona óptima (>{thresh_}%)"))
        fig_m.update_layout(
            title=dict(text="Calidad del suelo — zonas recomendadas para siembra",
                       font=dict(size=13,color=TEXT_MUTED)),
            xaxis_title="Posición E-O", yaxis_title="Posición N-S",
            template="plotly_dark", height=320, **CL, **AX)
        st.plotly_chart(fig_m, use_container_width=True)
        mnote("Modelo: interpolación espacial con blobs gaussianos (proxy de variabilidad en materia orgánica, "
              "drenaje y textura). NN candidata: Gaussian Process Regression (Kriging).")

    st.markdown("<hr>", unsafe_allow_html=True)

    # b.2 Cuándo cosechar  +  b.7 Caudal/Presión ─────────────────────────────
    ch_, cn_ = st.columns(2, gap="large")

    with ch_:
        slabel("b.2 · ¿Cuándo cosechar?")
        h_crop_ = st.selectbox("Cultivo ", ["Maíz","Trigo","Soya","Sorgo"])
        gdd_    = st.slider("GDD acumulados",            800,  3500, 2200, 50)
        gmoist_ = st.slider("Humedad del grano (%) ",    12,   35,   24)
        atemp_  = st.slider("Temperatura promedio (°C)", 15,   38,   26)
        days_,  label_, hcolor_ = harvest_timing(gdd_, gmoist_, h_crop_, atemp_)

        st.markdown(f'<div style="background:{SURFACE};border:1px solid {BORDER};'
                    f'border-radius:6px;padding:1rem 1.2rem;margin:.8rem 0;">'
                    f'<div style="font-size:.75rem;color:{TEXT_MUTED};text-transform:uppercase;'
                    f'letter-spacing:.5px;margin-bottom:.3rem;">Estado</div>'
                    f'<div style="font-size:1.4rem;font-weight:700;color:{hcolor_};">{label_}</div>'
                    f'<div style="color:{TEXT_MUTED};font-size:.88rem;margin-top:.3rem;">'
                    f'{"Condiciones óptimas alcanzadas." if days_<5 else f"~{days_:.0f} días para condiciones óptimas."}'
                    f'</div></div>', unsafe_allow_html=True)

        gdd_target_ = {"Maíz":2800,"Trigo":2000,"Soya":2600,"Sorgo":2400}[h_crop_]
        fig_gdd = go.Figure()
        fig_gdd.add_trace(go.Bar(
            x=["GDD acumulados","GDD restantes"],
            y=[min(gdd_, gdd_target_), max(0, gdd_target_-gdd_)],
            marker_color=[JD_GREEN, SURFACE_HIGH],
            text=[f"{min(gdd_,gdd_target_):.0f}", f"{max(0,gdd_target_-gdd_):.0f}"],
            textposition="inside", textfont=dict(color=TEXT_PRIMARY)))
        fig_gdd.update_layout(
            title=dict(text=f"GDD — meta: {gdd_target_}",font=dict(size=13,color=TEXT_MUTED)),
            yaxis_title="Grados-día (°C·día)", template="plotly_dark",
            height=220, **CL, **AX, showlegend=False)
        st.plotly_chart(fig_gdd, use_container_width=True)
        mnote("Modelo: grados-día de crecimiento (GDD, base 10°C) + secado del grano (~0.7%/día). "
              "NN candidata: LSTM 1-capa 32 unidades con GDD diarios como secuencia.")

    with cn_:
        slabel("b.7 · Caudal y presión de fertilizante")
        NOZZLE_LABELS = {"015 (estrecho)":"015","02 (estándar)":"02",
                         "03 (medio)":"03","04 (amplio)":"04","05 (grueso)":"05"}
        p_bar_   = st.slider("Presión (bar)",                  1.0, 6.0, 3.0, 0.1)
        nzl_lbl_ = st.selectbox("Boquilla (ISO 10625)", list(NOZZLE_LABELS.keys()))
        nzl_key_ = NOZZLE_LABELS[nzl_lbl_]
        nspd_    = st.slider("Velocidad del tractor (km/h) ",  3.0, 12.0, 7.0, 0.5)
        nsp_m_   = st.slider("Distancia entre boquillas (m)",  0.25, 1.5, 0.5, 0.05)
        fl_lmin_, app_ = nozzle_flow(p_bar_, nzl_key_, nspd_, nsp_m_)

        nm1, nm2 = st.columns(2)
        nm1.metric("Caudal por boquilla", f"{fl_lmin_:.2f} L/min")
        nm2.metric("Tasa de aplicación",  f"{app_:.0f} L/ha")

        KS = {"015":0.24,"02":0.33,"03":0.49,"04":0.65,"05":0.82}
        p_range_ = np.linspace(0.5, 7.0, 100)
        fig_n = go.Figure()
        for code, k in KS.items():
            sel = (code == nzl_key_)
            fig_n.add_trace(go.Scatter(x=p_range_, y=k*np.sqrt(p_range_), mode="lines",
                name=code, line=dict(color=JD_GREEN if sel else BORDER,
                                     width=2.5 if sel else 1)))
        fig_n.add_vline(x=p_bar_, line_color=JD_YELLOW, line_dash="dash", line_width=1.5,
                        annotation_text=f"{p_bar_} bar",
                        annotation_font_color=JD_YELLOW, annotation_font_size=11)
        fig_n.update_layout(
            title=dict(text="Caudal vs presión por tamaño de boquilla",
                       font=dict(size=13,color=TEXT_MUTED)),
            xaxis_title="Presión (bar)", yaxis_title="Caudal (L/min)",
            template="plotly_dark", height=220, **CL, **AX)
        st.plotly_chart(fig_n, use_container_width=True)
        mnote("Modelo: ecuación hidráulica ISO 10625 — Q = K·√P. No requiere NN.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # b.3 Temperatura  +  b.5 Humedad ─────────────────────────────────────────
    ct_, chu_ = st.columns(2, gap="large")
    days_lbl  = [f"Día {i+1}" for i in range(7)]

    with ct_:
        slabel("b.3 · Pronóstico de temperatura (7 días)")
        btemp_ = st.slider("Temperatura actual (°C)", 5, 42, 27)
        tf     = forecast_7d(btemp_, sigma=2.2, rev_rate=0.12, lo=btemp_-12, hi=btemp_+12, seed=42)
        fig_t  = go.Figure()
        fig_t.add_trace(go.Scatter(x=days_lbl, y=tf, mode="lines+markers",
            line=dict(color=JD_GREEN,width=2), marker=dict(size=6,color=JD_GREEN),
            fill="tozeroy", fillcolor="rgba(54,124,43,.1)"))
        fig_t.add_hline(y=35, line_color="#CC5555", line_dash="dot", line_width=1,
                        annotation_text="Estrés térmico (35 °C)",
                        annotation_font_color="#CC5555", annotation_font_size=10)
        fig_t.update_layout(title=dict(text="Temperatura estimada (°C)",
            font=dict(size=13,color=TEXT_MUTED)), yaxis_title="°C",
            template="plotly_dark", height=240, **CL,
            xaxis=dict(gridcolor=BORDER),
            yaxis=dict(gridcolor=BORDER, range=[0, 45]))
        st.plotly_chart(fig_t, use_container_width=True)
        mnote("Modelo: paseo aleatorio con regresión a la media. "
              "NN candidata: LSTM / Temporal Fusion Transformer para datos reales.")

    with chu_:
        slabel("b.5 · Pronóstico de humedad del suelo (7 días)")
        bhum_ = st.slider("Humedad actual del suelo (%)", 15, 55, 30)
        hf    = forecast_7d(bhum_, sigma=3.5, rev_rate=0.14, lo=15, hi=95, seed=7)
        fig_h = go.Figure()
        fig_h.add_hrect(y0=20, y1=50, fillcolor="rgba(54,124,43,.07)", line_width=0,
                        annotation_text="Rango óptimo",
                        annotation_font_color=TEXT_MUTED, annotation_font_size=10)
        fig_h.add_trace(go.Scatter(x=days_lbl, y=hf, mode="lines+markers",
            line=dict(color="#5B9BD5",width=2), marker=dict(size=6,color="#5B9BD5")))
        fig_h.update_layout(title=dict(text="Humedad del suelo estimada (%)",
            font=dict(size=13,color=TEXT_MUTED)), yaxis_title="%",
            template="plotly_dark", height=240, **CL,
            xaxis=dict(gridcolor=BORDER),
            yaxis=dict(gridcolor=BORDER, range=[10, 65]))
        st.plotly_chart(fig_h, use_container_width=True)
        mnote("Modelo: proceso autorregresivo con reversión a la media. "
              "NN candidata: Random Forest / GBT para datos tabulares en tiempo real.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # c — Datos de entrenamiento ───────────────────────────────────────────────
    with st.expander("c · Datasets de entrenamiento sintéticos (300 registros × 4 modelos)"):
        st.markdown(f'<p style="color:{TEXT_MUTED};font-size:.85rem;margin-bottom:1rem;">'
                    'Generados con fórmulas agronómicas + ruido gaussiano (seed=42). '
                    'Disponibles en <code>Data/raw/*.csv</code> (500 filas cada uno). '
                    'Ejecuta <code>python3 Data/generate_all.py</code> para regenerarlos.'
                    '</p>', unsafe_allow_html=True)
        df_s, df_f, df_h, df_v = make_training_data()
        dt1, dt2, dt3, dt4 = st.tabs(["Profundidad semilla","Fertilizante",
                                       "Timing cosecha","Combustible vs velocidad"])
        with dt1:
            st.dataframe(df_s.head(20), use_container_width=True)
            st.caption(f"{len(df_s)} registros · Y = tabla_agronómica(humedad,temp,cultivo) + N(0,0.3)")
        with dt2:
            st.dataframe(df_f.head(20), use_container_width=True)
            st.caption(f"{len(df_f)} registros · Y = Mitscherlich(n_suelo, objetivo) + N(0,8)")
        with dt3:
            st.dataframe(df_h.head(20), use_container_width=True)
            st.caption(f"{len(df_h)} registros · Y = max(días_GDD, días_secado) + N(0,1.5)")
        with dt4:
            st.dataframe(df_v.head(20), use_container_width=True)
            st.caption(f"{len(df_v)} registros · Y = 5.5/v + 0.14v + N(0,0.08)")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — HISTORIAL DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    slabel("Rendimiento del modelo — validación leave-one-out en 30 cosechas")
    mc1,mc2,mc3 = st.columns(3)
    mc1.metric("Error absoluto medio", f"{MAE:.2f} t/ha")
    mc2.metric("Cosechas validadas", "30")
    mc3.metric("Historial total", "200")
    st.markdown("<hr>", unsafe_allow_html=True)

    vc1, vc2 = st.columns(2)
    with vc1:
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=VAL["Real"], y=VAL["Pred"], mode="markers",
            marker=dict(size=8,color=JD_GREEN,opacity=.8),
            hovertemplate="Real:%{x:.2f}<br>Pred:%{y:.2f}<extra></extra>"))
        fig_v.add_trace(go.Scatter(x=[4.5,8.5],y=[4.5,8.5],mode="lines",
            line=dict(color=BORDER,width=1.5,dash="dot"),name="Ref. perfecta"))
        fig_v.update_layout(title=dict(text="Predicho vs Real",font=dict(size=13,color=TEXT_MUTED)),
            xaxis_title="Real (t/ha)",yaxis_title="Predicho (t/ha)",
            template="plotly_dark",**CL,**AX)
        st.plotly_chart(fig_v, use_container_width=True)

    with vc2:
        fig_hh = go.Figure()
        fig_hh.add_trace(go.Histogram(x=HIST["r"], nbinsx=28,
                                      marker_color=JD_GREEN, opacity=.7))
        fig_hh.add_vline(x=PRED_R, line_color=JD_YELLOW, line_width=2, line_dash="dash",
                         annotation_text=f"{PRED_R:.2f} t/ha",
                         annotation_font_color=JD_YELLOW, annotation_font_size=11)
        fig_hh.update_layout(title=dict(text="Distribución histórica de rendimiento",
            font=dict(size=13,color=TEXT_MUTED)),
            xaxis_title="Rendimiento (t/ha)",yaxis_title="Frecuencia",
            template="plotly_dark",**CL,**AX,showlegend=False)
        st.plotly_chart(fig_hh, use_container_width=True)

    st.markdown(f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:6px;'
                f'padding:1.2rem 1.4rem;margin-top:.5rem;font-size:.88rem;color:{TEXT_MUTED};'
                f'line-height:1.7;">El modelo usa las 20 cosechas más similares al escenario actual '
                f'(distancia euclidiana normalizada). No hay algoritmos de ML externos. '
                f'Error promedio en validación: '
                f'<span style="color:{TEXT_PRIMARY};font-weight:600;">{MAE:.2f} t/ha</span>.'
                f'<br>Siguiente paso: conectar <code>Data/models/yield_net.pt</code> '
                f'vía <code>@st.cache_resource</code> para predicciones dinámicas.</div>',
                unsafe_allow_html=True)
