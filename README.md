# Reto AgroCebada 2026 · Barley yield forecast

Forecast of the 2025 barley yield (t/ha) for the 59 blind parcels of Reto AgroCebada 2026 (Hidalgo, Puebla, Tlaxcala).

**Web page:** https://manuel04012005-star.github.io/agrocebada-2025/

Team: María Lucía López Ochoa · Manuel Sotos Baños · Valeria Peregrina De Sampedro · Ricardo Bañuelos Jesus

## Contents
- `index.html` — interactive results page (map, variables, forecast table).
- `resultados/entrega_59_parcelas.csv` — deliverable: `ID_POLIGONO, RENDIMIENTO_T_HA`.
- `resultados/predicciones_59_parcelas.csv` — forecasts with 80 % intervals.
- `codigo/` — Python pipeline:
  1. `features.py` builds 285 variables per parcel (Sentinel‑2, Landsat, Planet, CHIRPS/CHIRTS climate, INEGI CEM topography). Uses `shp.py` and `geotif.py` (pure‑Python shapefile and GeoTIFF readers).
  2. `modelo.py --cv-only` compares models using all variables.
  3. `modelo_final.py` trains the final model (Extra Trees, 30 variables + yield of the 5 nearest parcels) and writes the forecasts.
  4. `build_web.py` fills `plantilla.html` with the results to produce the page.

## Final model (5‑fold CV × 5 repeats, 138 training parcels)
| RMSE (t/ha) | MAE (t/ha) | R² |
|---|---|---|
| 0.492 | 0.355 | 0.663 |

Requirements: Python 3, pandas, numpy, scikit‑learn, matplotlib, tifffile. The challenge's input data are not included in this repository.
