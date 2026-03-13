# ☀ SolarIQ v2 — Climate-Aware Renewable Productivity Prediction System

> Complete full-stack solar energy prediction dashboard using BiLSTM deep learning + NASA POWER climate data

---

## 📁 Project Structure

```
solariq-v2/
├── frontend/
│   └── index.html          ← Complete dashboard (zero build, open in browser)
├── backend/
│   ├── main.py             ← FastAPI — all 9 phases implemented
│   └── requirements.txt
└── README.md
```

---

## ⚡ Quick Start (Zero Setup)

**Just open `frontend/index.html` in Chrome or Firefox.** No server needed.

The entire BiLSTM simulation, all 9 pipeline phases, charts, and exports run entirely in the browser.

---

## 🚀 Full Stack Deployment

### 1. Install backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`

### 2. Serve frontend
```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```

---

## 📡 API Reference

### `GET /locations`
Returns all 16 predefined NASA POWER solar sites with baseline GHI data.

### `POST /predict`
Run full 9-phase BiLSTM pipeline for two locations.

**Request:**
```json
{
  "locationA": {"name": "Khavda", "lat": 23.85, "lon": 69.72},
  "locationB": {"name": "Gajuwaka", "lat": 17.69, "lon": 83.22}
}
```

**Response includes:**
- Phase 1: `base_ghi`, `base_temp`, `base_wind`, `humidity`, `cloud_cover`
- Phase 2: `anomaly_score`
- Phase 4: BiLSTM model outputs
- Phase 5: `model_rmse`, `model_mae`, `model_r2`
- Phase 6: `forecast.years[]`, `forecast.ghi[]`, `forecast.temp[]`, `forecast.wind[]`
- Phase 7: `avg_ghi`, `eff_ghi`, `annual_kwh`, `variability`, `monthly_ghi[]`
- Phase 8: `risk_score`, `confidence`, `roi_years`, `npv_usd`, `lcoe`, `capex_usd`
- Phase 9: `recommendation` object with full reasoning

---

## 🧠 Pipeline Phases

| Phase | Description | Key Output |
|-------|-------------|-----------|
| 1 | Load NASA POWER data (GHI, T2M, WS10M, RH2M, CLOUD) | Climate baseline |
| 2 | Feature engineering: seasonal factors, anomaly score | `anomaly_score` |
| 3 | Multivariate time-series windowing for LSTM | Prepared sequences |
| 4 | BiLSTM training (2-layer, 128 units, dropout=0.2) | Trained model |
| 5 | Evaluation: RMSE, MAE, R² on holdout set | Model metrics |
| 6 | 20-year GHI / Temperature / Wind forecast (2026–2045) | Time series |
| 7 | Energy productivity: effective GHI, annual kWh, variability | Generation estimate |
| 8 | Risk & economics: risk score, ROI, NPV, LCOE | Investment metrics |
| 9 | Location comparison + final recommendation | Winner + reason |

---

## 📊 Dashboard Sections

1. **Pipeline Tracker** — Animated Phase 1–9 progress with status dots
2. **Location Selection** — 3 methods: search + geocoding, preset dropdown, map click
3. **KPI Cards** — Peak GHI, Annual Energy, Climate Risk, Model Confidence (with descriptions)
4. **Model Evaluation** — Phase 5: RMSE, MAE, R² per location with explanations
5. **GHI Forecast 2026–2045** — Line chart with climate trend insight
6. **Monthly GHI Distribution** — Seasonal bar chart per location
7. **Annual Energy Bar Chart** — Production comparison with financial insight
8. **Climate Risk Radar** — 5-axis: humidity, cloud, temp stress, variability, risk
9. **Temperature Trend** — Rising temperature projection with panel efficiency impact
10. **Wind Speed Trend** — Wind projection with cooling effect analysis
11. **Full Comparison Table** — 16 metrics side-by-side with best highlighted
12. **Economic Analysis** — CapEx, ROI, NPV, LCOE per location (Phase 8)
13. **Final Recommendation** — AI decision with full technical reasoning (Phase 9)
14. **Exports** — PDF, JSON, CSV download

---

## 🌍 Predefined Solar Sites (NASA POWER Baselines)

| Location | State | GHI (kWh/m²/d) | Base Temp |
|----------|-------|----------------|-----------|
| Radha Nesda | Gujarat | 6.80 | 29.5°C |
| Pugal | Rajasthan | 7.10 | 31.0°C |
| **Khavda** | Gujarat | **7.30** | 30.8°C |
| **Pang** | Ladakh | **7.50** | 15.2°C |
| Kurnool | Andhra Pradesh | 6.40 | 32.5°C |
| Pavagada | Karnataka | 6.60 | 31.0°C |
| Ananthapuramu | Andhra Pradesh | 6.50 | 32.0°C |
| Unchahar | Uttar Pradesh | 5.40 | 28.5°C |
| Ramagundam | Telangana | 5.80 | 33.0°C |
| Imphal Valley | Manipur | 4.90 | 24.0°C |
| Shillong | Meghalaya | 4.20 | 18.5°C |
| Kohima | Nagaland | 4.50 | 20.0°C |
| Itanagar | Arunachal Pradesh | 4.30 | 22.0°C |
| Guwahati | Assam | 4.60 | 26.5°C |
| Tirumani | Karnataka | 6.20 | 29.5°C |
| **Gajuwaka** | **Andhra Pradesh** | **6.10** | 31.5°C |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla JS + CSS Variables (no build) |
| Maps | Leaflet.js + OpenStreetMap |
| Charts | Chart.js 4.4 (Line, Bar, Radar) |
| Geocoding | OpenStreetMap Nominatim API |
| Fonts | Playfair Display + Outfit + IBM Plex Mono |
| Backend | Python FastAPI + Uvicorn |
| Data | NASA POWER satellite climate dataset |
| Model | BiLSTM simulation (mirrors Colab notebook) |

---

## 📄 Recommended for Hackathons

- **IIT AI Sprint**: Present the Phase 8 economic analysis + LCOE comparison
- **NASA Space Apps**: Highlight NASA POWER dataset usage + climate change integration
- **Research papers**: BiLSTM RMSE/MAE/R² metrics + 20-year forecast methodology
