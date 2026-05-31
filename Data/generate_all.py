"""
Generate all synthetic training datasets for AgroIntel models.
Run once: python3 generate_all.py
Outputs land in Data/raw/*.csv (500 rows each, seed=42).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path(__file__).parent / "raw"
RAW.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
N = 500
date_index = pd.date_range("2020-01-01", "2026-12-31", periods=N)
day_angle = 2 * np.pi * (date_index.dayofyear.to_numpy() / 365.25)
temp_aire = (24 + 9 * np.sin(day_angle - 0.7) + rng.normal(0, 2.2, N)).clip(4, 42)
precipitacion_mm = (24 + 22 * np.sin(day_angle + 1.1) + rng.normal(0, 14, N)).clip(0, 140)
humedad_relativa = (78 - 0.35 * (temp_aire - 24) - 0.08 * precipitacion_mm + rng.normal(0, 6, N)).clip(25, 98)
campo_id = rng.integers(1, 9, N)


# ── Dataset 1: Yield & cost (main Tab 1 model) ────────────────────────────────
humedad_grano = rng.normal(17, 3, N).clip(10, 30)
velocidad     = rng.normal(5.0, 1.2, N).clip(2, 8)
fertilizante  = rng.normal(130, 40, N).clip(0, 300)
pesticida     = rng.normal(2.5, 0.8, N).clip(0, 10)
temperatura   = rng.normal(27, 6, N).clip(5, 45)
humedad_aire  = rng.normal(62, 15, N).clip(20, 95)
horas_motor   = rng.normal(1400, 600, N).clip(0, 5000)
temp_motor    = rng.normal(84, 8, N).clip(60, 120)

base = np.full(N, 6.5)
base -= np.maximum(0, (humedad_grano - 14) * 0.12)
base -= np.maximum(0, (velocidad - 6.5) * 0.25)
base += np.minimum(0.9, fertilizante / 300 * 1.4)
base -= np.maximum(0, (temperatura - 35) * 0.08)
base -= np.maximum(0, (temp_motor - 95) * 0.03)
base += rng.normal(0, 0.4, N)
rendimiento_real = base.clip(1.5, 12.0)

costo_real = (
    fertilizante * 0.85
    + pesticida * 12.5
    + horas_motor * 0.35
    + rng.normal(0, 15, N)
)

df_yield = pd.DataFrame({
    "fecha":            date_index,
    "maquina":          "Cosechadora",
    "operacion":        "Cosecha",
    "campo_id":         campo_id,
    "temp_aire_c":      temp_aire,
    "precipitacion_mm": precipitacion_mm,
    "humedad_relativa_pct": humedad_relativa,
    "humedad_grano":    humedad_grano,
    "velocidad":        velocidad,
    "fertilizante":     fertilizante,
    "pesticida":        pesticida,
    "temperatura":      temperatura,
    "humedad_aire":     humedad_aire,
    "horas_motor":      horas_motor,
    "temp_motor":       temp_motor,
    "rendimiento_real": rendimiento_real,
    "costo_real":       costo_real,
})
df_yield.to_csv(RAW / "yield_harvest.csv", index=False)
print(f"yield_harvest.csv       {len(df_yield)} rows  MAE-floor ~0.40 t/ha")


# ── Dataset 2: Seed depth (b.1) ───────────────────────────────────────────────
soil_moist  = rng.uniform(10, 40, N)
soil_temp   = rng.uniform(5, 35, N)
crop_idx    = rng.integers(0, 4, N)
crops       = ["Maíz", "Trigo", "Soya", "Sorgo"]
base_depths = np.array([3.8, 3.0, 3.5, 2.5])[crop_idx]
moist_opt   = np.array([22,  18,  20,  20])[crop_idx]
temp_thresh = np.array([14,   9,  16,  19])[crop_idx]

depth = base_depths + (moist_opt - soil_moist) * 0.06
depth[soil_temp < temp_thresh + 4] -= 0.5
depth += rng.normal(0, 0.3, N)
depth = depth.clip(1.5, 8.0)

df_seed = pd.DataFrame({
    "fecha":            date_index,
    "maquina":          "Plantadora",
    "operacion":        "Siembra",
    "campo_id":         campo_id,
    "temp_aire_c":      temp_aire,
    "precipitacion_mm": precipitacion_mm,
    "humedad_relativa_pct": humedad_relativa,
    "humedad_suelo_pct": soil_moist,
    "temp_suelo_c":      soil_temp,
    # label-encode crops for model training; raw name for readability
    "cultivo":           [crops[i] for i in crop_idx],
    "cultivo_enc":       crop_idx.astype(float),
    "semillas_miles_ha":  (72 + crop_idx * 12 - (soil_moist - 22) * 0.35 + rng.normal(0, 4, N)).clip(40, 180),
    "profundidad_cm":    depth,
})
df_seed.to_csv(RAW / "seed_depth.csv", index=False)
print(f"seed_depth.csv          {len(df_seed)} rows")


# ── Dataset 3: Fertilizer N dose (b.4) ───────────────────────────────────────
n_soil_ppm   = rng.uniform(5, 60, N)
yield_target = rng.uniform(4, 10, N)
stages       = ["Siembra", "Vegetativo", "Reproductivo", "Madurez"]
stage_idx    = rng.integers(0, 4, N)
fractions    = np.array([0.30, 0.45, 0.20, 0.05])[stage_idx]

ymax          = yield_target * 1.18
c             = 0.008
n_soil_kgha   = n_soil_ppm * 3.9
n_total_needed = np.maximum(0, -np.log(1 - yield_target / ymax) / c)
n_applied     = np.maximum(0, n_total_needed - n_soil_kgha) * fractions
n_applied    += rng.normal(0, 8, N)
n_applied     = n_applied.clip(0, 280)

df_fert = pd.DataFrame({
    "fecha":          date_index,
    "maquina":        "Fertilizadora",
    "operacion":      "Fertilizacion",
    "campo_id":       campo_id,
    "temp_aire_c":    temp_aire,
    "precipitacion_mm": precipitacion_mm,
    "humedad_relativa_pct": humedad_relativa,
    "n_suelo_ppm":    n_soil_ppm,
    "objetivo_t_ha":  yield_target,
    "etapa":          [stages[i] for i in stage_idx],
    "etapa_enc":      stage_idx.astype(float),
    "dosis_kg_ha":    n_applied,
})
df_fert.to_csv(RAW / "fertilizer.csv", index=False)
print(f"fertilizer.csv          {len(df_fert)} rows")


# ── Dataset 4: Harvest timing (b.2) ──────────────────────────────────────────
gdd_curr     = rng.uniform(800, 3500, N)
grain_moist  = rng.uniform(12, 35, N)
avg_temp     = rng.uniform(15, 38, N)
crop_idx2    = rng.integers(0, 4, N)
gdd_targets  = np.array([2800, 2000, 2600, 2400])[crop_idx2]
moist_target = np.array([18,   14,   14,   16])[crop_idx2]

gdd_rem  = np.maximum(0, gdd_targets - gdd_curr)
daily_gdd = np.maximum(0.1, avg_temp - 10)
days_gdd  = gdd_rem / daily_gdd
days_dry  = np.maximum(0, grain_moist - moist_target) / 0.7
days_harv = np.maximum(days_gdd, days_dry) + rng.normal(0, 1.5, N)
days_harv = days_harv.clip(0, 90)

df_harvest = pd.DataFrame({
    "fecha":             date_index,
    "maquina":           "Cosechadora",
    "operacion":         "Planificacion cosecha",
    "campo_id":          campo_id,
    "temp_aire_c":       temp_aire,
    "precipitacion_mm":  precipitacion_mm,
    "humedad_relativa_pct": humedad_relativa,
    "gdd_acumulados":     gdd_curr,
    "humedad_grano_pct":  grain_moist,
    "temp_promedio_c":    avg_temp,
    "cultivo":            [crops[i] for i in crop_idx2],
    "cultivo_enc":        crop_idx2.astype(float),
    "dias_para_cosechar": days_harv,
})
df_harvest.to_csv(RAW / "harvest_timing.csv", index=False)
print(f"harvest_timing.csv      {len(df_harvest)} rows")


# ── Dataset 5: Speed vs fuel (a.2) ───────────────────────────────────────────
velocidad_s   = rng.uniform(3, 12, N)
horas_motor_s = rng.uniform(0, 5000, N)
# Tractor diesel curve: minimum near 6.3 km/h
fuel_lkm      = 5.5 / velocidad_s + 0.14 * velocidad_s + rng.normal(0, 0.08, N)
fuel_lkm      = fuel_lkm.clip(0.8, 4.5)

df_speed = pd.DataFrame({
    "fecha":            date_index,
    "maquina":          "Tractor",
    "operacion":        "Transporte",
    "campo_id":         campo_id,
    "temp_aire_c":      temp_aire,
    "precipitacion_mm": precipitacion_mm,
    "humedad_relativa_pct": humedad_relativa,
    "velocidad_kmh":    velocidad_s,
    "horas_motor":      horas_motor_s,
    "combustible_lkm":  fuel_lkm,
})
df_speed.to_csv(RAW / "speed_fuel.csv", index=False)
print(f"speed_fuel.csv          {len(df_speed)} rows")


print("\nAll datasets written to Data/raw/")
print("Next: cd training && python3 train_yield.py")
