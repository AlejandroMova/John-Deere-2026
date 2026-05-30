# AgroIntel — John Deere Pre-Harvest Decision Support Tool

**Reto 03 · Datos · John Deere × Tec de Monterrey**

A pre-harvest scenario simulator that lets a farmer input current field and machine conditions and receive a yield and cost prediction backed by historical harvest data — with full explainability down to the individual past records used.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Agronomic Rationale

### Why these variables?

| Variable | Agronomic basis |
|---|---|
| **Humedad del grano (%)** | Grain moisture at harvest is the single largest quality lever. High moisture forces slower harvesting speeds, raises drying costs, and increases field losses from header problems. The sweet spot for corn is 14–17 %; below 14 % risks excessive field dry-down losses; above 20 % signals the crop is too early. |
| **Velocidad de cosecha (km/h)** | Combine ground speed is the primary throughput control. Speeds above 6.5 km/h overload the threshing cylinder and rotor, increasing grain damage and unthreshed ears (MOG losses). Speed reduction of 1 km/h can cut grain loss by 0.2–0.4 ton/ha in high-yield conditions. |
| **Fertilizante (kg/ha)** | Nitrogen and phosphorus availability are the primary agronomic yield drivers. The formula applies a saturating benefit curve (`min(0.9, N/300 × 1.4)`) reflecting diminishing returns: the first 100 kg/ha give the largest yield bump; beyond ~193 kg/ha the yield response flattens. |
| **Pesticida (L/ha)** | Fungicide and herbicide inputs protect the yield from pest and disease pressure. In the synthetic model they affect only operational cost (not yield directly), since their effectiveness is already captured in the final grain condition and grain moisture observed at harvest. |
| **Temperatura ambiente (°C)** | Temperatures above 35 °C during grain fill accelerate kernel abortion and reduce test weight. Each degree above 35 °C costs approximately 0.08 ton/ha in this model, consistent with field research on corn heat stress. |
| **Humedad del aire (%)** | Relative humidity affects grain dry-down rate during the final ripening stage. It also interacts with combine separator efficiency (chaff stickiness). In the historical dataset it acts as a correlation proxy for regional climatic conditions. |
| **Horas de motor acumuladas** | Machine age is a proxy for component wear on the cylinder, rotor, and cleaning shoe — components that directly affect grain separation efficiency and grain damage rate. High accumulated hours correlate with higher maintenance costs captured in `costo_real`. |
| **Temperatura del motor (°C)** | Engine temperature above 95 °C signals thermal stress on the hydraulic and powertrain systems. Each degree above the threshold costs ~0.03 ton/ha in effective throughput reduction due to automatic deration from the ECU. |

### Yield formula derivation

```
rendimiento_real = 6.5              # regional baseline (ton/ha)
  − max(0, (humedad_grano − 14) × 0.12)   # moisture penalty
  − max(0, (velocidad − 6.5)   × 0.25)   # speed penalty  
  + min(0.9, fertilizante/300  × 1.40)   # fertilizer benefit (saturating)
  − max(0, (temperatura − 35)  × 0.08)   # heat stress penalty
  − max(0, (temp_motor − 95)   × 0.03)   # engine stress penalty
  + N(0, 0.4)                            # field-level stochastic variation
```

The noise term (σ = 0.4 ton/ha) represents unmodelled factors: micro-topography, soil variability, seed genetics, and operator skill.

### Cost formula derivation

```
costo_real = fertilizante × 0.85    # input cost ($0.85 / kg applied)
           + pesticida    × 12.50   # crop protection ($12.50 / L)
           + horas_motor  × 0.35    # machine depreciation + fuel ($0.35 / hr)
           + N(0, 15)               # fuel price variation and field overhead
```

---

## Model Architecture

### Why K-Nearest-Neighbors similarity search?

The app deliberately avoids black-box ML (random forests, XGBoost) for three reasons:

1. **Explainability**: Every prediction traces back to 20 real (or synthetic-but-realistic) past harvests a farmer can inspect.
2. **Data efficiency**: KNN needs no training phase and degrades gracefully with small datasets.
3. **Trust**: Showing the actual similar records in the scatter chart lets the operator validate the logic ("yes, those conditions look like mine").

### Distance normalization

All features are scaled to [0, 1] before computing Euclidean distance:

```
normalized = (value − feature_min) / (feature_max − feature_min)
```

Without normalization, features with large ranges (e.g., `horas_motor` 0–5000) would dominate features with small ranges (e.g., `pesticida` 0–10), ignoring agronomically important signals.

### Confidence score

```
confianza = clip(100 − mean_distance × 80, 40, 98)
```

High confidence means the 20 retrieved harvests are very close to the current input in normalized feature space — the historical record contains genuinely similar conditions. Low confidence means the prediction extrapolates from dissimilar conditions.

---

## Replacing Synthetic Data with Real John Deere Operations Center Data

### Data sources available via Operations Center API

| Synthetic column | Operations Center source | Notes |
|---|---|---|
| `humedad_grano` | Harvest Moisture (Grain Moisture Sensor) | Available per field pass via Machine Insights |
| `velocidad` | Ground Speed (CAN bus telemetry) | Logged per second; aggregate to field-level mean |
| `fertilizante` | Field Operations → Applied Nutrients | From as-applied map exports |
| `pesticida` | Field Operations → Applied Products | Combine with rate and area |
| `temperatura` | Weather tile overlay (JD integration) or on-machine thermometer | |
| `humedad_aire` | Weather API (OpenWeather or DTN) keyed to field lat/lon | |
| `horas_motor` | Machine → Engine Hours | From Equipment telemetry endpoint |
| `temp_motor` | Machine → Engine Coolant Temp | From Machine Insights telemetry |
| `rendimiento_real` | Yield Map → Field Summary → Dry Yield (ton/ha) | Requires yield mapping activated |
| `costo_real` | Financial Records → Field Cost Report | Or compute from input invoices |

### API integration sketch

```python
import requests

JD_BASE = "https://partnerapi.deere.com/platform"
HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Accept": "application/vnd.deere.axiom.v3+json"}

def fetch_field_operations(organization_id: str, field_id: str) -> dict:
    """Retrieve as-applied maps and harvest summaries for a field."""
    url = f"{JD_BASE}/organizations/{organization_id}/fields/{field_id}/fieldOperations"
    return requests.get(url, headers=HEADERS).json()

def fetch_machine_telemetry(machine_id: str, start: str, end: str) -> list[dict]:
    """Pull engine hours and temperature telemetry for a date range."""
    url = f"{JD_BASE}/machines/{machine_id}/telemetry"
    params = {"startDate": start, "endDate": end}
    return requests.get(url, headers=HEADERS, params=params).json().get("values", [])
```

After collecting per-harvest records, load them into the app by replacing the `generate_historical_data()` function with one that reads from your database:

```python
@st.cache_data(ttl=3600)          # refresh hourly
def load_historical_data() -> pd.DataFrame:
    return pd.read_parquet("s3://your-bucket/harvests.parquet")
```

No other changes to the app are needed — the prediction engine is data-agnostic.

### Minimum viable real dataset

The KNN model is useful with as few as **50 historical harvests** and becomes robust around 200+. With a live Operations Center connection, a single operation with 3–5 combines over 2–3 seasons easily reaches that threshold.

---

## File Structure

```
app.py            — single-file Streamlit application
requirements.txt  — Python dependencies
README.md         — this file
```
