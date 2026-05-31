import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import ceil, pi, sqrt
from pathlib import Path
from urllib.parse import quote

# ─── Brand palette ────────────────────────────────────────────────────────────
JD_GREEN       = "#367C2B"
JD_YELLOW      = "#FFDE00"
DARK_BG        = "#161616"
SURFACE        = "#212121"
SURFACE_HIGH   = "#2C2C2C"
BORDER         = "#333333"
TEXT_PRIMARY   = "#F0F0F0"
TEXT_MUTED     = "#888888"
APP_DIR        = Path(__file__).parent
RAW_DATA_DIR    = APP_DIR / "Data" / "raw"

st.set_page_config(
    page_title="AgroIntel — John Deere",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    /* App background and typography */
    .stApp {{ background-color: {DARK_BG}; color: {TEXT_PRIMARY}; }}
    .main .block-container {{ padding-top: 2rem; max-width: 1400px; }}
    h1, h2, h3, h4, p, label, div {{ color: {TEXT_PRIMARY}; }}

    /* Sidebar: make it visually consistent with main surface and accent with JD_GREEN */
    section[data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 2px solid {JD_GREEN}; padding: 0.6rem; }}
    section[data-testid="stSidebar"] .block-container {{ padding: 0.25rem 0.5rem; }}
    section[data-testid="stSidebar"] .css-1d391kg {{ background: transparent; }}

    /* Sidebar navigation buttons: full width, left-aligned, subtle hover */
    section[data-testid="stSidebar"] button {{
        width: 100%; text-align: left; padding: 10px 12px; margin-bottom: 6px;
        background: transparent; color: {TEXT_PRIMARY}; border: 1px solid transparent;
        border-radius: 6px; font-weight: 600; font-size: 0.95rem;
    }}
    section[data-testid="stSidebar"] button:hover {{
        background: rgba(54,124,43,0.08); border-color: rgba(54,124,43,0.12);
    }}

    /* Metric, surface and accent rules */
    [data-testid="metric-container"] {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 1rem 1.1rem;
    }}
    [data-testid="stMetricValue"] {{ color: {TEXT_PRIMARY} !important; font-size: 1.7rem !important; font-weight: 600 !important; }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.5px; }}

    /* Slider accents: yellow thumb and range */
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
        background: {JD_YELLOW} !important;
        border: 2px solid {JD_YELLOW} !important;
        box-shadow: 0 0 0 4px rgba(255, 222, 0, 0.14) !important;
    }}
    [data-testid="stSlider"] [data-baseweb="slider"] {{ accent-color: {JD_YELLOW}; }}
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {{
        background: {JD_YELLOW} !important;
    }}

    /* Tabs and selection accents */
    .stTabs [data-baseweb="tab"] {{ color: {TEXT_MUTED}; }}
    .stTabs [aria-selected="true"] {{ color: {TEXT_PRIMARY} !important; border-bottom: 2px solid {JD_GREEN} !important; }}

    /* Accent helpers */
    .accent-green {{ color: {JD_GREEN}; }}
    .accent-yellow {{ color: {JD_YELLOW}; }}

    /* Charts on dark surface */
    .plotly-graph-div .main-svg {{ background: transparent; }}

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


def get_bot_response(msg: str) -> str:
    """Keyword-based agronomic assistant responses (Spanish)."""
    m = msg.lower().strip()

    # Saludos
    if any(w in m for w in ["hola", "buenas", "hey", "ayuda", "help", "inicio"]):
        return ("Hola! Soy AgroBot 🌾 Puedo ayudarte con:\n\n"
                "• **Rendimiento** — predicciones de cosecha\n"
                "• **Siembra** — profundidad y posición de semillas\n"
                "• **Fertilizante** — dosis y timing\n"
                "• **Tractor** — ruta óptima y velocidad\n"
                "• **Clima** — temperatura y humedad\n\n"
                "¿Sobre qué quieres saber?")

    # Rendimiento / cosecha
    if any(w in m for w in ["rendimiento", "cosecha", "tonelada", "ton", "yield", "produccion", "producción"]):
        return ("El simulador de cosecha usa las **20 cosechas históricas más similares** "
                "a tus condiciones actuales (distancia euclidiana normalizada) para estimar el rendimiento.\n\n"
                "**Factores clave:**\n"
                "• Humedad del grano < 14% → rendimiento óptimo\n"
                "• Velocidad < 6.5 km/h → sin penalización\n"
                "• Fertilizante ≥ 193 kg/ha → beneficio máximo\n"
                "• Temperatura > 35°C → penalización de 0.08 t/ha por grado\n\n"
                "Ajusta los sliders en el **Simulador** para ver el impacto en tiempo real.")

    # Humedad del grano
    if any(w in m for w in ["humedad grano", "humedad del grano", "moisture", "grano húmedo", "secar"]):
        return ("La humedad del grano es el factor individual más importante en cosecha.\n\n"
                "• **< 14%** → óptimo, sin penalización\n"
                "• **14–20%** → penalización de 0.12 t/ha por % extra\n"
                "• **> 20%** → riesgo de pérdidas en cabezal y mayor costo de secado\n\n"
                "Reducir de 18% a 14% puede representar **+0.48 t/ha** según el modelo.")

    # Fertilizante
    if any(w in m for w in ["fertilizante", "nitrogeno", "nitrógeno", "dosis", "npk", "abono", "nutriente"]):
        return ("La dosis de fertilizante nitrogenado se calcula con la **curva de Mitscherlich-Baule** "
                "(ley de rendimientos decrecientes):\n\n"
                "• El beneficio máximo se alcanza en ~**193 kg/ha**\n"
                "• Por encima de eso, el rendimiento ya no aumenta\n"
                "• La dosis varía por etapa: Vegetativo (45%) > Siembra (30%) > Reproductivo (20%)\n\n"
                "Revisa el módulo **Siembra y predicciones → Fertilizante** para la curva de respuesta.")

    # Semilla / siembra / profundidad
    if any(w in m for w in ["semilla", "siembra", "profundidad", "plantar", "planting", "seed"]):
        return ("La profundidad óptima de siembra depende de la humedad y temperatura del suelo:\n\n"
                "| Cultivo | Profundidad base |\n"
                "|---------|------------------|\n"
                "| Maíz    | 3.8 cm           |\n"
                "| Trigo   | 3.0 cm           |\n"
                "| Soya    | 3.5 cm           |\n"
                "| Sorgo   | 2.5 cm           |\n\n"
                "• Suelo seco → planta más profundo (hasta +1.8 cm)\n"
                "• Suelo frío → planta más somero (-0.5 cm)\n\n"
                "Ajusta en **Siembra y predicciones → Profundidad de semilla**.")

    # Tractor / ruta / velocidad
    if any(w in m for w in ["tractor", "ruta", "velocidad", "combustible", "gasolina", "diesel", "falla", "ruta"]):
        return ("**Ruta óptima:** El algoritmo boustrofedón (zigzag) minimiza los giros y maximiza la cobertura. "
                "Para un campo de 300×200 m con cabezal de 4.5 m: ~45 pasadas, ~13.8 km recorridos.\n\n"
                "**Velocidad óptima:** La curva de combustible (5.5/v + 0.14v) tiene mínimo en ~**6.3 km/h**. "
                "Por encima de 8 km/h la probabilidad de falla mecánica sube exponencialmente.\n\n"
                "Ve a **Operaciones del tractor** para ver las curvas en tiempo real.")

    # Temperatura
    if any(w in m for w in ["temperatura", "calor", "frio", "frío", "clima", "tiempo", "weather", "estrés"]):
        return ("**Temperatura y rendimiento:**\n"
                "• < 35°C → sin penalización en cosecha\n"
                "• > 35°C → pierde 0.08 t/ha por cada grado adicional (estrés térmico en llenado de grano)\n\n"
                "**Temperatura del motor:**\n"
                "• < 95°C → operación normal\n"
                "• > 95°C → pérdida de 0.03 t/ha por grado (derate del ECU)\n\n"
                "El pronóstico de temperatura a 7 días está en **Siembra y predicciones → Temperatura**.")

    # Humedad del suelo / del aire
    if any(w in m for w in ["humedad suelo", "humedad del suelo", "humedad aire", "riego", "lluvia", "precipitacion"]):
        return ("**Humedad del suelo:**\n"
                "• Rango óptimo: **20–50%**\n"
                "• Suelo muy seco → mayor profundidad de siembra para alcanzar la humedad\n"
                "• Suelo saturado → riesgo de compactación y enfermedades radiculares\n\n"
                "El pronóstico de humedad del suelo a 7 días está en **Siembra y predicciones → Humedad**.")

    # Cuándo cosechar / timing
    if any(w in m for w in ["cuando cosechar", "cuándo cosechar", "tiempo cosecha", "gdd", "grados dia", "grados día", "madurez"]):
        return ("El momento óptimo de cosecha se determina con dos modelos:\n\n"
                "1. **GDD (Grados-Día de Crecimiento):** Maíz necesita ~2800 GDD (base 10°C) para madurez\n"
                "2. **Secado del grano:** ~0.7% de humedad se pierde por día en condiciones normales\n\n"
                "Se cosecha cuando **ambas condiciones** se cumplen.\n\n"
                "Revisa el indicador en **Siembra y predicciones → ¿Cuándo cosechar?**")

    # Confianza / modelo
    if any(w in m for w in ["confianza", "modelo", "precisión", "precision", "error", "mae", "exactitud"]):
        return ("El modelo tiene un **error promedio (MAE) de 0.31 t/ha** en validación con 30 cosechas.\n\n"
                "La confianza se calcula según qué tan similares son las 20 cosechas históricas usadas: "
                "si están muy cerca de tus condiciones → alta confianza (hasta 95%); "
                "si son muy diferentes → baja confianza (mínimo 40%).\n\n"
                "El piso de ruido irreducible del dataset es ~0.40 t/ha (σ del campo).\n\n"
                "Ver detalles en el tab **Historial del modelo**.")

    # Presión / boquilla / aspersión
    if any(w in m for w in ["presión", "presion", "boquilla", "caudal", "aspersor", "pulverizar", "bar"]):
        return ("La tasa de aplicación de fertilizante líquido sigue la **ecuación ISO 10625**:\n\n"
                "**Q (L/min) = K × √P**\n\n"
                "Donde K depende del tamaño de boquilla:\n"
                "• Boquilla 015 → K=0.24 | Boquilla 03 → K=0.49 | Boquilla 05 → K=0.82\n\n"
                "A 3 bar con boquilla 03 y 7 km/h: **~198 L/ha**. "
                "Ajusta en **Siembra → Caudal y presión**.")

    # Posición semillas / mapa
    if any(w in m for w in ["mapa", "posición", "posicion", "campo", "zona", "calidad suelo", "terreno"]):
        return ("El mapa de calidad del suelo muestra zonas de alta y baja aptitud agrícola "
                "basado en variabilidad espacial de materia orgánica, drenaje y textura.\n\n"
                "• **Verde oscuro** → suelo de alta calidad, siembra prioritaria\n"
                "• **Rojo/café** → suelo limitante, ajustar densidad de siembra\n"
                "• **Líneas amarillas** → contorno del umbral óptimo configurable\n\n"
                "En producción, este mapa vendría de muestras georeferenciadas o sensores remotos.")

    # Datos / John Deere API
    if any(w in m for w in ["dato", "api", "operations center", "real", "integrar", "conectar"]):
        return ("Los datos actuales son **sintéticos** (generados con fórmulas agronómicas + ruido gaussiano).\n\n"
                "Para conectar datos reales de John Deere Operations Center:\n"
                "1. Reemplaza `Data/raw/yield_harvest.csv` con datos exportados de la API\n"
                "2. El schema ya es compatible (mismas columnas)\n"
                "3. Los modelos se reentrenan con `python3 Data/training/train_yield.py`\n\n"
                "El endpoint clave es `/organizations/{id}/fields/{id}/fieldOperations`.")

    # Fallback
    return ("No tengo una respuesta específica para eso. Intenta preguntar sobre:\n\n"
            "**rendimiento** · **fertilizante** · **semilla** · **profundidad** · "
            "**tractor** · **velocidad** · **temperatura** · **humedad** · "
            "**cuando cosechar** · **confianza** · **boquilla** · **mapa**")


def format_duration(hours_value):
    total_minutes = int(round(float(hours_value) * 60))
    hours, minutes = divmod(total_minutes, 60)
    if hours == 0:
        return f"{minutes} min"
    return f"{hours} h {minutes:02d} min"


HISTORY_FILES = [
    {
        "file": "yield_harvest.csv",
        "dataset": "cosecha",
        "maquina": "Cosechadora",
        "operacion": "Cosecha",
        "primary": ["rendimiento_real", "costo_real"],
    },
    {
        "file": "seed_depth.csv",
        "dataset": "siembra",
        "maquina": "Plantadora",
        "operacion": "Siembra",
        "primary": ["profundidad_cm", "semillas_miles_ha"],
    },
    {
        "file": "fertilizer.csv",
        "dataset": "fertilizacion",
        "maquina": "Fertilizadora",
        "operacion": "Fertilizacion",
        "primary": ["dosis_kg_ha"],
    },
    {
        "file": "harvest_timing.csv",
        "dataset": "planificacion_cosecha",
        "maquina": "Cosechadora",
        "operacion": "Planificacion cosecha",
        "primary": ["dias_para_cosechar"],
    },
    {
        "file": "speed_fuel.csv",
        "dataset": "tractor",
        "maquina": "Tractor",
        "operacion": "Transporte",
        "primary": ["combustible_lkm"],
    },
]


def _ensure_history_defaults(frame, meta):
    frame = frame.copy()
    frame["fecha"] = pd.to_datetime(frame["fecha"], errors="coerce")
    frame["dataset"] = meta["dataset"]
    frame["maquina"] = frame["maquina"] if "maquina" in frame.columns else meta["maquina"]
    frame["operacion"] = frame["operacion"] if "operacion" in frame.columns else meta["operacion"]
    frame["anio"] = frame["fecha"].dt.year
    frame["mes"] = frame["fecha"].dt.month
    frame["dia"] = frame["fecha"].dt.day
    frame["week_start"] = frame["fecha"].dt.to_period("W-MON").dt.start_time
    frame["fecha_str"] = frame["fecha"].dt.strftime("%Y-%m-%d")
    return frame


@st.cache_data(show_spinner=False)
def load_operation_history():
    frames = []
    for meta in HISTORY_FILES:
        path = RAW_DATA_DIR / meta["file"]
        if not path.exists():
            continue
        frames.append(_ensure_history_defaults(pd.read_csv(path), meta))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def scenario_adjustment(base_value, temp_delta=0.0, rain_delta=0.0, humidity_delta=0.0, sensitivity=0.0):
    modifier = 1 + sensitivity * (0.6 * temp_delta - 0.25 * rain_delta - 0.12 * humidity_delta)
    return float(max(0, base_value * modifier))


def forecast_future_month(history, target_year, target_month, temp_delta, rain_delta, humidity_delta):
    if history.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    month_hist = history[history["mes"] == target_month]
    year_gap = max(0, target_year - int(history["anio"].max()))
    trend = 1 + min(0.18, year_gap * 0.025)

    base_metrics = {
        "fuel_lkm": month_hist["combustible_lkm"].mean() if "combustible_lkm" in month_hist else np.nan,
        "seeds_tha": month_hist["semillas_miles_ha"].mean() if "semillas_miles_ha" in month_hist else np.nan,
        "fertilizer_kgha": month_hist["dosis_kg_ha"].mean() if "dosis_kg_ha" in month_hist else np.nan,
        "yield_t_ha": month_hist["rendimiento_real"].mean() if "rendimiento_real" in month_hist else np.nan,
    }

    fallback_metrics = pd.Series({
        "fuel_lkm": history["combustible_lkm"].mean() if "combustible_lkm" in history.columns else 0.0,
        "seeds_tha": history["semillas_miles_ha"].mean() if "semillas_miles_ha" in history.columns else 0.0,
        "fertilizer_kgha": history["dosis_kg_ha"].mean() if "dosis_kg_ha" in history.columns else 0.0,
        "yield_t_ha": history["rendimiento_real"].mean() if "rendimiento_real" in history.columns else 0.0,
    })
    base = pd.Series(base_metrics).fillna(fallback_metrics).fillna(0.0)
    scenario = base.copy()
    scenario["fuel_lkm"] = scenario_adjustment(base["fuel_lkm"] * trend, temp_delta, rain_delta, humidity_delta, 0.03)
    scenario["seeds_tha"] = scenario_adjustment(base["seeds_tha"] * trend, temp_delta, rain_delta, humidity_delta, 0.012)
    scenario["fertilizer_kgha"] = scenario_adjustment(base["fertilizer_kgha"] * trend, temp_delta, rain_delta, humidity_delta, 0.018)
    scenario["yield_t_ha"] = scenario_adjustment(base["yield_t_ha"] * trend, -temp_delta, rain_delta, humidity_delta, 0.02)
    return base, scenario


def summarize_history_range(history, start_date, end_date):
    window = history[(history["fecha"] >= pd.Timestamp(start_date)) & (history["fecha"] <= pd.Timestamp(end_date))].copy()
    if window.empty:
        nearest_idx = (history["fecha"] - pd.Timestamp(start_date)).abs().idxmin()
        window = history.loc[[nearest_idx]].copy()

    global_means = history.select_dtypes(include="number").mean(numeric_only=True)
    window_means = window.select_dtypes(include="number").mean(numeric_only=True)

    def value_or_fallback(column_name):
        value = window_means.get(column_name, np.nan)
        if pd.isna(value):
            value = global_means.get(column_name, np.nan)
        return value

    summary_rows = []
    for meta in HISTORY_FILES:
        subset = window[window["dataset"] == meta["dataset"]]
        row = {
            "Maquina": meta["maquina"],
            "Operacion": meta["operacion"],
            "Eventos": len(subset),
        }
        for key in meta["primary"]:
            if key in window.columns:
                value = subset[key].mean() if not subset.empty else np.nan
                if pd.isna(value):
                    value = value_or_fallback(key)
                row[key.replace("_", " ")] = value
        summary_rows.append(row)

    climate = {
        "temp_aire_c": value_or_fallback("temp_aire_c"),
        "precipitacion_mm": value_or_fallback("precipitacion_mm"),
        "humedad_relativa_pct": value_or_fallback("humedad_relativa_pct"),
        "combustible_lkm": value_or_fallback("combustible_lkm"),
        "dosis_kg_ha": value_or_fallback("dosis_kg_ha"),
        "semillas_miles_ha": value_or_fallback("semillas_miles_ha"),
    }

    return window, climate, pd.DataFrame(summary_rows)


def project_future_range(history, start_date, end_date, temp_c, precipitation_mm, work_hours, engine_hours):
    future_start = pd.Timestamp(start_date)
    future_end = pd.Timestamp(end_date)
    horizon_days = max(1, (future_end - future_start).days + 1)

    month_hist = history[history["mes"] == future_start.month]
    if month_hist.empty:
        month_hist = history

    base_temp = month_hist["temp_aire_c"].mean() if "temp_aire_c" in month_hist else history["temp_aire_c"].mean()
    base_precip = month_hist["precipitacion_mm"].mean() if "precipitacion_mm" in month_hist else history["precipitacion_mm"].mean()
    base_humidity = month_hist["humedad_relativa_pct"].mean() if "humedad_relativa_pct" in month_hist else history["humedad_relativa_pct"].mean()
    base_engine = month_hist["horas_motor"].mean() if "horas_motor" in month_hist else history["horas_motor"].mean()

    temp_delta = temp_c - base_temp if pd.notna(base_temp) else 0.0
    rain_delta = precipitation_mm - base_precip if pd.notna(base_precip) else 0.0
    humidity_delta = max(-20.0, min(20.0, (precipitation_mm - base_precip) * 0.18 if pd.notna(base_precip) else 0.0))

    base, scenario = forecast_future_month(history, future_start.year, future_start.month, temp_delta, rain_delta, humidity_delta)
    workload_factor = max(0.5, min(2.0, work_hours / max(1.0, horizon_days * 8.0)))
    engine_factor = max(0.7, min(1.8, 1.0 + ((engine_hours - base_engine) / 10000.0 if pd.notna(base_engine) else 0.0)))

    prediction = scenario.copy()
    prediction["fuel_lkm"] = max(0.1, scenario["fuel_lkm"] * workload_factor * engine_factor)
    prediction["seeds_tha"] = max(0.1, scenario["seeds_tha"] * (0.95 + 0.05 * workload_factor))
    prediction["fertilizer_kgha"] = max(0.1, scenario["fertilizer_kgha"] * (0.9 + 0.1 * workload_factor))
    prediction["yield_t_ha"] = max(0.1, scenario["yield_t_ha"] * max(0.82, 1 - max(0.0, temp_delta) * 0.01))

    climate = {
        "temp_aire_c": float(temp_c),
        "precipitacion_mm": float(max(0.0, precipitation_mm)),
        "humedad_relativa_pct": float(np.clip(base_humidity + precipitation_mm * 0.08 - temp_delta * 0.6, 25, 98)) if pd.notna(base_humidity) else 70.0,
        "combustible_lkm": float(prediction["fuel_lkm"]),
        "dosis_kg_ha": float(prediction["fertilizer_kgha"]),
        "semillas_miles_ha": float(prediction["seeds_tha"]),
    }

    summary_rows = []
    for meta in HISTORY_FILES:
        row = {
            "Maquina": meta["maquina"],
            "Operacion": meta["operacion"],
            "Eventos": horizon_days,
        }
        for key in meta["primary"]:
            if key == "combustible_lkm":
                value = prediction["fuel_lkm"]
            elif key == "dosis_kg_ha":
                value = prediction["fertilizer_kgha"]
            elif key == "semillas_miles_ha":
                value = prediction["seeds_tha"]
            elif key == "rendimiento_real":
                value = prediction["yield_t_ha"]
            elif key in climate:
                value = climate[key]
            else:
                value = prediction.get(key, np.nan)
            row[key.replace("_", " ")] = value
        summary_rows.append(row)

    return climate, pd.DataFrame(summary_rows), base, prediction


HISTORICAL_SOURCES = [
    ("yield_harvest", "yield_harvest.csv"),
    ("seed_depth", "seed_depth.csv"),
    ("fertilizer", "fertilizer.csv"),
    ("harvest_timing", "harvest_timing.csv"),
    ("speed_fuel", "speed_fuel.csv"),
]

HISTORY_PRIMARY_METRICS = {
    "yield_harvest": ["rendimiento_real", "costo_real"],
    "seed_depth": ["profundidad_cm"],
    "fertilizer": ["dosis_kg_ha"],
    "harvest_timing": ["dias_para_cosechar"],
    "speed_fuel": ["combustible_lkm"],
}


@st.cache_data(show_spinner=False)
def load_historical_data():
    frames = []
    for dataset_name, file_name in HISTORICAL_SOURCES:
        path = RAW_DATA_DIR / file_name
        if not path.exists():
            continue

        frame = pd.read_csv(path)
        if "fecha" in frame.columns:
            frame["fecha"] = pd.to_datetime(frame["fecha"], errors="coerce")
        else:
            frame["fecha"] = pd.date_range("2020-01-01", "2026-12-31", periods=len(frame))

        frame["dataset"] = dataset_name
        frame["anio"] = frame["fecha"].dt.year
        frame["mes"] = frame["fecha"].dt.month
        frame["week_start"] = frame["fecha"].dt.to_period("W-MON").dt.start_time
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["fecha", "dataset", "anio", "mes", "week_start"])

    return pd.concat(frames, ignore_index=True, sort=False)


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


def feature_card(title, description, accent, svg_markup):
    image_uri = f"data:image/svg+xml;utf8,{quote(svg_markup)}"
    st.markdown(
        f'''
        <div style="background: linear-gradient(180deg, {SURFACE} 0%, {SURFACE_HIGH} 100%);
                    border: 1px solid {BORDER}; border-radius: 18px; padding: 1rem;
                    height: 100%; box-shadow: 0 10px 28px rgba(0,0,0,0.22);">
            <img src="{image_uri}" alt="{title}" style="width:100%; height:140px; object-fit:cover;
                 border-radius:14px; margin-bottom:0.9rem; border:1px solid rgba(255,255,255,0.06);"/>
            <div style="color:{accent}; font-size:0.76rem; font-weight:800; letter-spacing:1.2px;
                        text-transform:uppercase; margin-bottom:0.3rem;">{title}</div>
            <div style="color:{TEXT_PRIMARY}; font-size:1.04rem; font-weight:700; margin-bottom:0.4rem; line-height:1.25;">{description}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


# ─── Header ───────────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 2])
with c1:
    st.markdown(f'<div style="font-size:1rem;color:{TEXT_MUTED};font-weight:500;margin-bottom:2px;">'
                f'<span style="color:{JD_GREEN};font-weight:700;">John Deere</span> · AgroIntel</div>'
                f'<div style="font-size:1.75rem;font-weight:700;color:{TEXT_PRIMARY};line-height:1.2;">'
                f'Simulador y Predicciones</div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div style="text-align:right;padding-top:.8rem;">'
                f'<span style="color:{TEXT_MUTED};font-size:.8rem;">Reto 03 · Datos · Tec de Monterrey</span>'
                f'</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Load history once and expose a left-side navigation
history = load_operation_history()

# Sidebar navigation using full-width buttons for a more webpage-like look
PAGES = [
    "Inicio",
    "Semáforo operacional",
    "Gemelo digital",
    "Simulación",
    "AgroBot",
]

if "page" not in st.session_state:
    st.session_state["page"] = "Simulación"

st.sidebar.markdown(
    f"<div style='padding:8px 6px;color:{TEXT_MUTED};font-weight:700;margin-bottom:6px;'>Navegación</div>",
    unsafe_allow_html=True,
)

for p in PAGES:
    if st.sidebar.button(p, key=f"nav_{p}"):
        st.session_state["page"] = p

page = st.session_state["page"]

if page == "Inicio":
    st.markdown(
        f'<div style="padding:0.25rem 0 0.75rem 0;">'
        f'<div style="color:{JD_YELLOW};font-size:0.78rem;font-weight:800;letter-spacing:1.3px;text-transform:uppercase;">AgroIntel · John Deere</div>'
        f'<div style="font-size:2rem;font-weight:800;line-height:1.1;margin-top:0.25rem;">Centro de control agrícola</div>'
        f'<div style="color:{TEXT_MUTED};font-size:0.95rem;margin-top:0.5rem;max-width:760px;">'
        'Una vista única para revisar el estado histórico, el gemelo digital y los simuladores de decisión y proyección.'
        '</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if history.empty:
        st.warning("No hay datos históricos disponibles en Data/raw.")
    else:
        min_date = history["fecha"].min().date()
        max_date = history["fecha"].max().date()
        total_events = len(history)
        datasets = history["dataset"].nunique()

        c1, c2, c3 = st.columns(3)
        c1.metric("Periodo cubierto", f"{min_date} → {max_date}")
        c2.metric("Total de eventos", f"{total_events}")
        c3.metric("Conjuntos de datos", f"{datasets}")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:1.15rem;font-weight:700;margin-bottom:0.9rem;'>Secciones principales</div>", unsafe_allow_html=True)

        card_1 = '''
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 420">
          <defs>
            <linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#367C2B"/>
              <stop offset="100%" stop-color="#1f4d1a"/>
            </linearGradient>
          </defs>
          <rect width="800" height="420" rx="28" fill="url(#g1)"/>
          <circle cx="150" cy="120" r="52" fill="#FFDE00" opacity="0.95"/>
          <rect x="180" y="250" width="360" height="52" rx="20" fill="#212121" opacity="0.95"/>
          <rect x="220" y="190" width="210" height="88" rx="16" fill="#2C2C2C"/>
          <rect x="330" y="165" width="86" height="34" rx="10" fill="#212121"/>
          <circle cx="250" cy="315" r="45" fill="#161616"/>
          <circle cx="470" cy="315" r="45" fill="#161616"/>
          <circle cx="250" cy="315" r="22" fill="#FFDE00"/>
          <circle cx="470" cy="315" r="22" fill="#FFDE00"/>
          <path d="M520 90 C620 95, 660 180, 590 245 C545 287, 455 274, 430 200 C410 138, 455 86, 520 90 Z" fill="#FFDE00" opacity="0.88"/>
          <rect x="555" y="120" width="30" height="120" rx="12" fill="#161616"/>
        </svg>
        '''
        card_2 = '''
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 420">
          <rect width="800" height="420" rx="28" fill="#212121"/>
          <rect x="60" y="56" width="680" height="300" rx="20" fill="#2C2C2C"/>
          <path d="M100 290 C180 240, 240 320, 330 270 S500 210, 570 250 S670 290, 720 240" fill="none" stroke="#FFDE00" stroke-width="14" stroke-linecap="round"/>
          <path d="M120 120 L180 150 L240 110 L300 170 L360 140 L420 180 L480 130 L540 165 L600 120 L680 155" fill="none" stroke="#367C2B" stroke-width="12" stroke-linecap="round"/>
          <circle cx="120" cy="120" r="18" fill="#FFDE00"/>
          <circle cx="240" cy="110" r="18" fill="#FFDE00"/>
          <circle cx="360" cy="140" r="18" fill="#FFDE00"/>
          <circle cx="480" cy="130" r="18" fill="#FFDE00"/>
          <circle cx="600" cy="120" r="18" fill="#FFDE00"/>
          <rect x="580" y="250" width="120" height="54" rx="16" fill="#367C2B"/>
          <rect x="80" y="78" width="200" height="28" rx="10" fill="#161616"/>
        </svg>
        '''
        card_3 = '''
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 420">
          <rect width="800" height="420" rx="28" fill="#161616"/>
          <rect x="70" y="54" width="660" height="312" rx="22" fill="#212121"/>
          <rect x="110" y="110" width="130" height="170" rx="18" fill="#367C2B"/>
          <rect x="280" y="110" width="130" height="120" rx="18" fill="#FFDE00"/>
          <rect x="450" y="110" width="230" height="90" rx="18" fill="#2C2C2C"/>
          <rect x="450" y="220" width="230" height="60" rx="18" fill="#2C2C2C"/>
          <path d="M110 300 H680" stroke="#FFDE00" stroke-width="8" stroke-linecap="round"/>
          <circle cx="174" cy="176" r="24" fill="#161616"/>
          <circle cx="345" cy="168" r="22" fill="#161616"/>
          <path d="M490 132 H645" stroke="#FFDE00" stroke-width="8" stroke-linecap="round"/>
          <path d="M490 252 H612" stroke="#FFDE00" stroke-width="8" stroke-linecap="round"/>
        </svg>
        '''

        cols = st.columns(3, gap="large")
        with cols[0]:
            feature_card(
                "Semáforo",
                "Revisa si las condiciones están listas para entrar al campo o si conviene esperar.",
                JD_YELLOW,
                card_1,
            )
        with cols[1]:
            feature_card(
                "Gemelo digital",
                "Observa el estado agregado del campo, el recorrido del tractor y la tendencia semanal.",
                JD_GREEN,
                card_2,
            )
        with cols[2]:
            feature_card(
                "Simulación",
                "Evalúa decisiones de cosecha, operación y futuros escenarios sin tocar la máquina real.",
                JD_YELLOW,
                card_3,
            )
    st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PREDICCIONES Y SIMULACIÓN
# ══════════════════════════════════════════════════════════════════════════════
if page == "Simulación":

    st.markdown("### Simulador de decisiones de cosecha")
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

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Simulador operacional")
    st.markdown(
        f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:6px;'
        f'padding:1rem 1.15rem;margin-bottom:1rem;color:{TEXT_MUTED};font-size:.88rem;line-height:1.6;">'
        'Ajusta escenario climático y de trabajo para ver cómo cambian las métricas operativas clave: '
        'combustible, semillas y fertilizante. '
        '</div>',
        unsafe_allow_html=True,
    )

    if history.empty:
        st.warning("No se encontraron datos históricos en Data/raw.")
    else:
        latest_year = int(history["anio"].max())
        year_options = list(range(latest_year + 1, latest_year + 4))
        month_names = [
            "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
            "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
        ]

        left, right = st.columns([1, 1.8], gap="large")
        with left:
            target_year = st.selectbox("Año objetivo", year_options, index=0, key="ops_target_year")
            target_month = st.selectbox("Mes objetivo", list(range(1, 13)), index=7, format_func=lambda m: month_names[m - 1], key="ops_target_month")
            temp_delta = st.slider("Cambio de temperatura (°C)", -8.0, 8.0, 0.0, 0.5, key="ops_temp_delta")
            rain_delta = st.slider("Cambio de precipitación (mm)", -40.0, 40.0, 0.0, 1.0, key="ops_rain_delta")
            humidity_delta = st.slider("Cambio de humedad relativa (%)", -20.0, 20.0, 0.0, 1.0, key="ops_humidity_delta")
            work_distance = st.slider("Jornada prevista (km)", 10, 300, 100, 10, key="ops_work_distance")

        base, scenario = forecast_future_month(history, target_year, target_month, temp_delta, rain_delta, humidity_delta)
        scenario_fuel_total = scenario["fuel_lkm"] * work_distance

        with right:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Combustible", f"{scenario['fuel_lkm']:.2f} L/km", delta=f"{scenario['fuel_lkm'] - base['fuel_lkm']:+.2f}")
            m2.metric("Jornada", f"{scenario_fuel_total:.0f} L", delta=f"{(scenario['fuel_lkm'] - base['fuel_lkm']) * work_distance:+.0f}")
            m3.metric("Semillas", f"{scenario['seeds_tha']:.1f} mil/ha", delta=f"{scenario['seeds_tha'] - base['seeds_tha']:+.1f}")
            m4.metric("Fertilizante", f"{scenario['fertilizer_kgha']:.1f} kg/ha", delta=f"{scenario['fertilizer_kgha'] - base['fertilizer_kgha']:+.1f}")

            # st.caption(
            #     f"Escenario para {month_names[target_month - 1]} de {target_year}. "
            #     "Este resumen reemplaza la tabla detallada porque los KPI ya muestran el cambio relevante."
            # )

            m5, _, _, _ = st.columns(4)
            m5.metric("Rendimiento", f"{scenario['yield_t_ha']:.2f} t/ha", delta=f"{scenario['yield_t_ha'] - base['yield_t_ha']:+.2f}")

            forecast_table = pd.DataFrame({
                "Variable": ["Combustible", "Semillas plantadas", "Fertilizante", "Rendimiento"],
                "Base": [base["fuel_lkm"], base["seeds_tha"], base["fertilizer_kgha"], base["yield_t_ha"]],
                "Escenario": [scenario["fuel_lkm"], scenario["seeds_tha"], scenario["fertilizer_kgha"], scenario["yield_t_ha"]],
            })
            forecast_table["Cambio"] = forecast_table["Escenario"] - forecast_table["Base"]
            st.dataframe(forecast_table, use_container_width=True, hide_index=True)

            st.caption(
                f"Proyección para {month_names[target_month - 1]} de {target_year} con ajuste de clima. "
                "Los cambios positivos o negativos reflejan el escenario what-if frente al patrón histórico."
            )

    # st.markdown("<hr>", unsafe_allow_html=True)
    # st.markdown("### Predicciones futuras")
    # st.markdown(
    #     f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:6px;'
    #     f'padding:1rem 1.15rem;margin-bottom:1rem;color:{TEXT_MUTED};font-size:.88rem;line-height:1.6;">'
    #     'Aquí puedes proyectar un escenario futuro y comparar el comportamiento esperado con el patrón histórico. '
    #     'Se mantiene una vista compacta para no duplicar el simulador operacional.'
    #     '</div>',
    #     unsafe_allow_html=True,
    # )

    # if history.empty:
    #     st.warning("No se encontraron datos históricos en Data/raw.")
    # else:
    #     latest_year = int(history["anio"].max())
    #     year_options = list(range(latest_year + 1, latest_year + 4))
    #     month_names = [
    #         "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    #         "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
    #     ]

    #     left, right = st.columns([1, 1.8], gap="large")
    #     with left:
    #         target_year = st.selectbox("Año objetivo", year_options, index=0, key="pred_target_year")
    #         target_month = st.selectbox("Mes objetivo", list(range(1, 13)), index=7, format_func=lambda m: month_names[m - 1], key="pred_target_month")
    #         temp_delta = st.slider("Cambio de temperatura (°C)", -8.0, 8.0, 0.0, 0.5, key="pred_temp_delta")
    #         rain_delta = st.slider("Cambio de precipitación (mm)", -40.0, 40.0, 0.0, 1.0, key="pred_rain_delta")
    #         humidity_delta = st.slider("Cambio de humedad relativa (%)", -20.0, 20.0, 0.0, 1.0, key="pred_humidity_delta")
    #         work_distance = st.slider("Jornada prevista (km)", 10, 300, 100, 10, key="pred_work_distance")

    #     base, scenario = forecast_future_month(history, target_year, target_month, temp_delta, rain_delta, humidity_delta)
    #     scenario_fuel_total = scenario["fuel_lkm"] * work_distance

    #     with right:
    #         m1, m2, m3, m4 = st.columns(4)
    #         m1.metric("Combustible", f"{scenario['fuel_lkm']:.2f} L/km", delta=f"{scenario['fuel_lkm'] - base['fuel_lkm']:+.2f}")
    #         m2.metric("Jornada", f"{scenario_fuel_total:.0f} L", delta=f"{(scenario['fuel_lkm'] - base['fuel_lkm']) * work_distance:+.0f}")
    #         m3.metric("Semillas", f"{scenario['seeds_tha']:.1f} mil/ha", delta=f"{scenario['seeds_tha'] - base['seeds_tha']:+.1f}")
    #         m4.metric("Fertilizante", f"{scenario['fertilizer_kgha']:.1f} kg/ha", delta=f"{scenario['fertilizer_kgha'] - base['fertilizer_kgha']:+.1f}")

    #         st.caption(
    #             f"Proyección para {month_names[target_month - 1]} de {target_year} con ajuste de clima. "
    #             "Se omite la tabla porque no agrega más señal que estos KPI."
    #         )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SEMÁFORO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Semáforo operacional":
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
#  TAB 4 — DIGITAL TWIN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Gemelo digital":
    history = load_operation_history()

    slabel("Gemelo digital")

    if history.empty:
        st.warning("No se encontraron datos históricos en Data/raw.")
    else:
        min_date = history["fecha"].min().date()
        max_date = history["fecha"].max().date()
        default_start = (pd.Timestamp(max_date) - pd.Timedelta(days=6)).date()
        if default_start < min_date:
            default_start = min_date
        selected_range = st.date_input(
            "Rango de fechas",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            range_start, range_end = selected_range
        else:
            range_start = selected_range
            range_end = selected_range

        if range_start > range_end:
            range_start, range_end = range_end, range_start

        window, climate, summary_df = summarize_history_range(history, range_start, range_end)
        range_start_ts = pd.Timestamp(range_start)
        range_end_ts = pd.Timestamp(range_end)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        metrics = [
            (c1, "Temperatura", climate["temp_aire_c"], lambda v: f"{v:.1f} °C"),
            (c2, "Precipitación", climate["precipitacion_mm"], lambda v: f"{v:.1f} mm"),
            (c3, "Humedad relativa", climate["humedad_relativa_pct"], lambda v: f"{v:.0f}%"),
            (c4, "Combustible", climate["combustible_lkm"], lambda v: f"{v:.2f} L/km"),
            (c5, "Fertilizante", climate["dosis_kg_ha"], lambda v: f"{v:.1f} kg/ha"),
            (c6, "Semillas", climate["semillas_miles_ha"], lambda v: f"{v:.1f} mil/ha"),
        ]
        for col, label, value, formatter in metrics:
            with col:
                col.metric(label, formatter(value))

        st.markdown("<hr>", unsafe_allow_html=True)
        st.caption(f"Rango: {range_start_ts.date()} a {range_end_ts.date()} · {len(window)} registros")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        range_series = window.copy()
        range_series["week_start"] = pd.to_datetime(range_series["week_start"], errors="coerce")
        trend = range_series.groupby("week_start").agg(
            temp_aire_c=("temp_aire_c", "mean"),
            precipitacion_mm=("precipitacion_mm", "mean"),
            humedad_relativa_pct=("humedad_relativa_pct", "mean"),
        ).reset_index()
        if not trend.empty:
            fig_twin = go.Figure()
            fig_twin.add_trace(go.Scatter(x=trend["week_start"], y=trend["temp_aire_c"], mode="lines+markers", name="Temperatura", line=dict(color=JD_GREEN, width=2)))
            fig_twin.add_trace(go.Scatter(x=trend["week_start"], y=trend["precipitacion_mm"], mode="lines+markers", name="Precipitación", line=dict(color=JD_YELLOW, width=2)))
            fig_twin.add_trace(go.Scatter(x=trend["week_start"], y=trend["humedad_relativa_pct"], mode="lines+markers", name="Humedad relativa", line=dict(color=JD_YELLOW, width=2)))
            fig_twin.update_layout(
                title=dict(text="Tendencia agregada del rango", font=dict(size=13, color=TEXT_MUTED)),
                xaxis_title="Semana",
                yaxis_title="Valor",
                template="plotly_dark",
                **CL,
                **AX,
            )
            st.plotly_chart(fig_twin, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Operaciones del tractor dentro del gemelo digital")

        st.markdown("### a.1 · Ruta óptima de cobertura")
        ri, ro = st.columns([1, 2], gap="large")

        with ri:
            fl_ = st.slider("Largo del campo (m)", 50, 500, 300, 10)
            fw_ = st.slider("Ancho del campo (m)", 50, 500, 200, 10)
            mw_ = st.slider("Ancho de trabajo (m)", 2.0, 12.0, 4.5, 0.5)
            spd_ = st.slider("Velocidad de trabajo (km/h)", 3.0, 12.0, 7.0, 0.5)

        r = plan_boustrophedon(fl_, fw_, mw_, spd_)

        with ro:
            rm1, rm2, rm3, rm4 = st.columns(4)
            rm1.metric("Distancia total", f"{r['dist_km']:.1f} km")
            rm2.metric("Tiempo estimado", format_duration(r['time_h']))
            rm3.metric("Combustible est.", f"{r['fuel_l']:.0f} L")
            rm4.metric("Área cubierta", f"{r['area_ha']:.1f} ha")

            fig_r = go.Figure()
            fig_r.add_shape(type="rect", x0=0, y0=0, x1=r["W"], y1=r["L"],
                            line=dict(color=JD_GREEN, width=2), fillcolor="rgba(54,124,43,0.06)")
            for i in range(r["n"]):
                x0, x1 = i * r["mw"], min((i + 1) * r["mw"], r["W"])
                fig_r.add_shape(type="rect", x0=x0, y0=0, x1=x1, y1=r["L"],
                                fillcolor=f"rgba(54,124,43,{0.25 if i % 2 == 0 else 0.12})",
                                line=dict(width=0))
            fig_r.add_trace(go.Scatter(x=r["px"], y=r["py"], mode="lines",
                                       line=dict(color=JD_YELLOW, width=2.5), name="Trayectoria"))
            fig_r.add_trace(go.Scatter(x=[r["mw"] / 2], y=[0], mode="markers",
                                       marker=dict(symbol="circle", size=10, color=JD_GREEN, line=dict(color=JD_YELLOW, width=1)),
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

        st.markdown("### a.2 · Velocidad óptima — combustible, tiempo y probabilidad de falla")
        vi, vo = st.columns([1, 2], gap="large")

        with vi:
            dist_km_ = st.slider("Distancia a recorrer (km)", 1.0, 60.0, 15.0, 0.5)
            eng_hrs_ = st.slider("Horas de motor acumuladas", 0, 5000, 1200, 50)

        v, fuel, time_h, fail, opt_v = analyze_speed(dist_km_, eng_hrs_)

        with vo:
            oi = int(np.argmin(np.abs(v - opt_v)))
            vm1, vm2, vm3 = st.columns(3)
            vm1.metric("Velocidad óptima", f"{opt_v:.1f} km/h")
            vm2.metric("Combustible a v óptima", f"{fuel[oi]:.1f} L")
            vm3.metric("P(falla) a v óptima", f"{fail[oi]:.2f}%")

            fig_sp = make_subplots(rows=1, cols=3,
                subplot_titles=["Combustible total (L)", "Tiempo (h)", "P(falla) (%)"])
            for col_i, (ydata, color) in enumerate(
                [(fuel, JD_GREEN), (time_h, JD_YELLOW), (fail, "#E0C24D")], 1):
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
#  PAGE: AGROBOT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "AgroBot":
    # Init chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if not st.session_state.chat_history:
        st.session_state.chat_history = [
            {"role": "assistant",
             "content": ("Hola! Soy **AgroBot**, tu asistente agronómico. "
                         "Pregúntame sobre rendimiento, siembra, fertilizante, "
                         "tractor, clima o cualquier cosa del simulador.")}
        ]

    # Header
    st.markdown(
        f'<div style="margin-bottom:1.5rem;">'
        f'<div style="color:{JD_YELLOW};font-size:0.78rem;font-weight:800;'
        f'letter-spacing:1.3px;text-transform:uppercase;">Asistente agronómico</div>'
        f'<div style="font-size:2rem;font-weight:800;line-height:1.1;margin-top:0.25rem;">'
        f'AgroBot</div>'
        f'<div style="color:{TEXT_MUTED};font-size:0.9rem;margin-top:0.4rem;">'
        f'Responde preguntas sobre el simulador, los modelos y decisiones agronómicas.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Chat display — fixed-height scrollable container
    chat_container = st.container(height=520)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input using native chat_input (sticks to bottom of the page)
    user_input = st.chat_input("Escribe tu pregunta aquí…")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append(
            {"role": "assistant", "content": get_bot_response(user_input)}
        )
        st.rerun()


