# 🚗 Used Car Price Predictor — India

An end-to-end ML project that predicts used car prices for the Indian market. Built with real scraped data, XGBoost, and SHAP-based explainability — not a Kaggle notebook demo.

> **Live demo:** *(deploy link here)*  
> **Tech:** Python · XGBoost · SHAP · Streamlit · CarDekho API

---

## What it does

Enter a car's details — brand, model, year, kilometres, fuel type, city — and get:

- **Fair market price estimate** with a conservative/optimistic negotiation band
- **SHAP waterfall chart** showing exactly which factors pushed the price up or down, and by how much
- **Market segment benchmark** placing the car in its price band (budget → luxury)
- **Plain-English explanations** per feature, written for buyers and sellers
- **Buyer & seller tips** derived directly from SHAP values

---

## Folder structure

```
used-car-price-predictor/
├── data/
│   ├── raw/                  # Scraped + Kaggle CSVs
│   └── processed/
│       └── cars_processed.csv
├── models/
│   ├── car_price_model.pkl   # Trained XGBoost model
│   ├── encoders.pkl          # LabelEncoders for Brand, Model, Location
│   └── brand_model_map.json  # {Brand: [Model, ...]} for UI dropdowns
├── notebooks/
│   └── eda_training.ipynb
├── src/
│   ├── preprocess.py
│   ├── predict.py
│   └── explainer.py
├── app/
│   └── app.py
└── requirements.txt
```

---

## Dataset

| Source | Rows | Notes |
|--------|------|-------|
| CarDekho API (live scraped, May 2026) | 305 | City-filtered, deduplicated |
| Kaggle CarDekho v4 | ~2,059 | `cars1.csv` |
| Kaggle CarDekho v3 | ~7,253 | `cars2.csv` |
| **Final (after cleaning)** | **4,287** | 12 brands · 31 models · 11 cities · 2004–2022 |

**Features:** `Brand`, `Km`, `Owner`, `Location`, `Engine`, `Seats`, `Base Model`, `Car Age`, `Km Per Year`, `Diesel`, `Petrol`, `Manual Transmission`

**Target:** `log1p(Price)` — trained on log scale, converted back with `expm1` at prediction time.

---

## Model performance

| Metric | Value |
|--------|-------|
| R² | **0.9519** |
| MAE | ₹86,446 |
| RMSE | ₹1,73,132 |
| MAPE | **10.31%** |

**Algorithm:** XGBoost Regressor, tuned with RandomizedSearchCV (50 iterations, 5-fold CV).

---

## Key engineering decisions

**Log-transform the target** — Price is right-skewed. `log1p(Price)` normalises the distribution and prevents expensive cars from dominating the loss function.

**Car Age instead of Year** — `Car_Age = 2024 - Year` directly encodes depreciation age, reducing redundancy and improving tree splits.

**Km_Per_Year as an engineered feature** — `Km / Car_Age` detects commercial/taxi use (>25,000 km/year). Raw Km has a weak -0.13 correlation with price; Km_Per_Year surfaces a cleaner signal.

**Model_Base = first word of Model** — Collapses 2,737 unique model strings to 31 base models. Case/hyphen variants (`EcoSport` / `Ecosport`) are deduplicated against a count-weighted canonical form.

**Label encoding for high-cardinality categoricals** — Brand, Model, and Location are label-encoded. XGBoost handles these correctly and avoids the dimensionality explosion of one-hot encoding across 12 brands × 31 models × 11 cities.

**Dropped Mileage, kept Engine** — Mileage only existed in some source files. Keeping it would create source-dependent NaN patterns acting as data leakage. Engine was available across enough rows to retain.

---

## Data engineering challenges

Three non-obvious bugs found and fixed during scraping.

**CarDekho city filter silently ignored** — The `searchstring` param was supposed to filter by city but returned a national feed dominated by Gurgaon regardless. Root cause: the API requires a numeric `cityId` param alongside the string filter. Before fix: 19,200 rows but only 48 unique listings (same 48 repeated 400×).

**Silent duplicate accumulation** — No memory of seen listings across paginated requests, silently inflating the dataset. Fix: global `seen_urls` set checked before appending each result.

**Hard per-query cap (~20 listings)** — CarDekho returns at most ~20 listings per search query, making city-only search yield only ~300 listings nationally. Fix: **model × city combo strategy** — 30 models × 15 cities = 450 unique queries, each returning a fresh non-overlapping set.

```
City-only search:       ~300 unique listings
Model × city strategy:  ~9,000 unique listings  (30× improvement)
```

---

## SHAP explainability

Every prediction is explained with SHAP TreeExplainer, structured for three audiences:

- **Buyers/sellers** — rupee impact cards with negotiation tips
- **Technical users** — SHAP values table with log-scale contributions
- **Visual** — waterfall chart showing the step-by-step path from dataset average to final prediction

Top features by mean |SHAP|: Car Age (~0.25), Engine (~0.25), Base Model (~0.09), Location (~0.08), Brand (~0.07). Car age and engine size together account for ~50% of explanatory power.

---

## Data limitations

- **2021–2025 cars are underrepresented** — predictions for very recent cars may underestimate market price
- **Dataset skews 2004–2019** — most reliable in this range
- **Listed prices, not transaction prices** — actual sale prices are typically 5–15% lower
- **11 cities only** — cars in unlisted cities are mapped to the nearest available city

---

## Running locally

```bash
git clone https://github.com/yourname/used-car-price-predictor.git
cd used-car-price-predictor
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

---

## Future work

- [ ] FastAPI wrapper for the prediction endpoint
- [ ] Comparable listings panel — 5 similar cars from the dataset beside each prediction
- [ ] Price trend chart by year for the selected model
- [ ] Periodic retraining with fresher CarDekho data