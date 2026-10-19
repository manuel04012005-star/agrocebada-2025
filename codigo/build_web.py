"""Inserts model results and EDA cross-checks into the template and writes the final page (English, matches the EDA deck)."""
import json, re, sys, pandas as pd, numpy as np
sys.path.insert(0, '/home/claude/proy/src'); import shp
O = '/home/claude/proy/out/'
F = pd.read_csv(O + 'features.csv', index_col=0)
R = json.load(open(O + 'resultados_final.json'))
pr = pd.read_csv(O + 'predicciones_59_parcelas.csv').set_index('ID_POLIGONO')
oof = pd.DataFrame(R['oof']).set_index('ID_POLIGONO')
dbf = shp.read_dbf().set_index('ID_POLIGONO')

# ---------------- model names in English ----------------
NAME = {'Línea base (media municipio)': 'Municipality mean (baseline)', 'Vecinos espaciales (kNN, k=5)': 'Nearest neighbors (kNN, k=5)',
        'ElasticNet': 'Elastic Net', 'HistGB': 'Hist. Gradient Boosting'}
def en_model(m):
    base, _, tail = m['modelo'].partition(' (')
    if m['modelo'] in NAME: name = NAME[m['modelo']]
    else: name = NAME.get(base, base)
    grp = {'Referencia': 'reference'}.get(m['grupo'], m['grupo'].replace('30 variables + vecinos', '30 variables + neighbors'))
    return dict(m, modelo=name if m['modelo'] in NAME else f"{name} ({grp})", grupo=grp)
met = [en_model(m) for m in R['metricas']]
FINAL = en_model(next(m for m in R['metricas'] if m['modelo'] == R['elegido']))['modelo']
BASE = NAME['Línea base (media municipio)']
m_best = next(m for m in met if m['modelo'] == FINAL); base = next(m for m in met if m['modelo'] == BASE)

# ---------------- variable labels ----------------
MES = {'04': 'Apr', '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov'}
FIX = {'log_area': 'Parcel area (log)', 'lon': 'Longitude', 'lat': 'Latitude', 'elev_media': 'Mean elevation', 'elev_std': 'Elevation spread',
       'pend_media': 'Mean slope', 'pend_std': 'Slope spread', 'cultivo_mayusculas': 'Record format (CEBADA / Cebada)',
       'prec_ciclo_2025': 'Rain Apr–Oct 2025', 'prec_jun_ago_2025': 'Rain Jun–Aug 2025', 'prec_sep_oct_2025': 'Rain Sep–Oct 2025',
       'prec_ciclo_hist': 'Rain Apr–Oct, 2022–24 mean', 'prec_anom_2025': 'Rain anomaly 2025 vs 2022–24', 'tmax_jun_sep_2025': 'Tmax Jun–Sep 2025',
       'tmin_jun_sep_2025': 'Tmin Jun–Sep 2025', 'tmed_jun_sep_2025': 'Tmean Jun–Sep 2025', 'amp_term_jun_sep_2025': 'Thermal range Jun–Sep 2025',
       'gdd_ciclo_2025': 'Degree days Apr–Oct 2025', 'tmax_anom_2025': 'Tmax anomaly 2025 vs 2022–24', 'hist_ndvi_max': 'Peak NDVI, 2022–24 mean',
       'hist_ndvi_auc': 'NDVI area under curve, 2022–24', 'hist_doy_max': 'Peak NDVI date, 2022–24', 'ndvi_max_vs_hist': 'Peak NDVI 2025 vs 2022–24',
       'rend_vecinos': "Yield of the 5 nearest parcels"}
FEN = {'max': 'peak', 'base': 'baseline', 'amp': 'amplitude', 'doy_max': 'peak date', 'doy_inicio': 'green‑up date', 'doy_fin': 'end of season',
       'duracion': 'season length', 'auc': 'area under curve', 'auc_total': 'total area', 'tasa_verdeo': 'green‑up rate'}
def label(v):
    if v in FIX: return FIX[v]
    m = re.match(r'prec_2025_(\d\d)$', v)
    if m: return f'Rain {MES.get(m[1], m[1])} 2025'
    if v.startswith('Estado_'): return 'State: ' + v[7:]
    if v.startswith('Municipio_'): return 'Municipality: ' + v[10:]
    m = re.match(r'(s2|ls|pl)_(\w+?)_m(\d\d)$', v)
    if m: return f"{ {'s2': 'S2', 'ls': 'Landsat', 'pl': 'Planet'}[m[1]]} {m[2].upper().replace('NDVI_STD', 'NDVI std')} {MES[m[3]]}"
    m = re.match(r'(s2|pl)_(ndvi|lai|fapar|ndwi)25_(\w+)$', v)
    if m: return f"{ {'s2': 'S2', 'pl': 'Planet'}[m[1]]} {m[2].upper()} 2025 {FEN.get(m[3], m[3])}"
    return v
for it in R['importancia']: it['etiqueta'] = label(it['var'])

# ---------------- parcels ----------------
parcelas = []
for pid, r in F.iterrows():
    d = dict(id=pid, estado=r.Estado, municipio=r.Municipio, conjunto=r.CONJUNTO, area=float(r.AREA_HA), lon=float(r.lon), lat=float(r.lat),
             elev=float(r.elev_media), pend=None if pd.isna(r.pend_media) else float(r.pend_media), prec=float(r.prec_ciclo_2025), ndvimax=float(r.s2_ndvi25_max))
    if r.CONJUNTO == 'ENTRENAMIENTO': d.update(real=float(r.RENDIMIENTO_T_HA), oof=float(oof.loc[pid, 'pred']))
    else: d.update(pred=float(pr.loc[pid, 'RENDIMIENTO_PRED_T_HA']), lo=float(pr.loc[pid, 'LIM_INF_80']), hi=float(pr.loc[pid, 'LIM_SUP_80']))
    parcelas.append(d)

# ---------------- EDA cross-checks (recomputed from the raw files) ----------------
UP = '/root/.claude/uploads/81985831-d14b-58e2-bb98-e4949aa24c45/'
b = pd.read_csv('/mnt/user-data/uploads/Conjunto_datos_BASICO_AgroCebada2026.csv', usecols=['ID_POLIGONO', 'fecha_captura', 'sensor', 'ndvi_promedio', 'vi6t_promedio', 'porcentaje_nubosidad'])
b['f'] = pd.to_datetime(b.fecha_captura, dayfirst=True)
miss_s2 = b[b.sensor == 'Sentinel-2'].ndvi_promedio.isna().mean(); miss_ls = b[b.sensor == 'Landsat'].vi6t_promedio.isna().mean()
s2 = b[(b.sensor == 'Sentinel-2') & (b.porcentaje_nubosidad <= 20)]
s2all = b[b.sensor == 'Sentinel-2']   # peak NDVI as in the EDA deck: all valid Sentinel-2 scenes, Apr–Oct
pk = s2all[s2all.f.dt.month.between(4, 10)].groupby([s2all.f.dt.year, 'ID_POLIGONO']).ndvi_promedio.max()
pkmed = pk.groupby(level=0).median()
T = F[F.CONJUNTO == 'ENTRENAMIENTO']
r_pk = np.corrcoef(pk.loc[2025].reindex(T.index), T.RENDIMIENTO_T_HA)[0, 1]
rounded = ((T.RENDIMIENTO_T_HA * 2) % 1 == 0)
fmt_low = dbf.loc[T.index, 'Cultivo'] == 'Cebada'
st_mean = T.groupby('Estado').RENDIMIENTO_T_HA.mean()
top = R['importancia'][0]
eda = [
    dict(slide='Slides 3–4 · Missing values & cloud cover', titulo='Cloud cover is almost binary',
         hallazgo=f"Images are either clear or fully cloudy, and the cloudy ones carry no index values: <b>{miss_s2:.0%}</b> of Sentinel‑2 rows and <b>{miss_ls:.0%}</b> of Landsat rows are empty.",
         decision='Scenes above 20 % cloud cover are dropped, the same threshold as in the EDA, and each sensor stays in its own variables.'),
    dict(slide='Slides 6 & 8 · NDVI by month', titulo='July–October is the window',
         hallazgo=f"NDVI greens up in July and peaks in Aug–Sep. Mean 2025 NDVI: Jul <b>0.47</b>, Aug <b>0.55</b>, Sep <b>0.56</b>, Oct <b>0.47</b>.",
         decision="Each parcel's 2025 curve gives phenology variables: peak, peak date, green‑up date, season length and area under the curve."),
    dict(slide='Slide 7 · Peak NDVI by year', titulo='Yearly conditions matter',
         hallazgo=f"Median peak NDVI: 2022 <b>{pkmed[2022]:.2f}</b> · 2023 <b>{pkmed[2023]:.2f}</b> · 2024 <b>{pkmed[2024]:.2f}</b> · 2025 <b>{pkmed[2025]:.2f}</b>. 2023 was the weakest season.",
         decision='Climate enters as 2025 rain and temperature plus their anomaly against 2022–24, and each parcel keeps its 2022–24 NDVI history.'),
    dict(slide='Slides 10–11 & 17 · Indices vs yield', titulo='Greenness alone does not predict yield',
         hallazgo=f"Peak NDVI vs yield gives <b>r ≈ {r_pk:.2f}</b>. Planet peak indices are also weak (−0.17 to −0.25).",
         decision=f"Zone, management and climate carry the signal. The most important variable in the final model is <b>{top['etiqueta'].lower()}</b>."),
    dict(slide='Slide 9 · Barley yield', titulo='Two groups, or two ways of recording',
         hallazgo=f"Yield is bimodal and <b>{rounded.mean():.0%}</b> of values are round (2.5, 3.0, 3.5). Puebla averages <b>{st_mean['Puebla']:.2f}</b> t/ha, Tlaxcala <b>{st_mean['Tlaxcala']:.2f}</b>.",
         decision=f"The shapefile's Cultivo field is written \"Cebada\" or \"CEBADA\". {rounded[fmt_low].mean():.0%} of \"Cebada\" yields are round versus {rounded[~fmt_low].mean():.0%} of \"CEBADA\" ones, so the record format became a variable."),
    dict(slide='Slides 10, 17–18 · Correlations & pairplot', titulo='The indices are redundant',
         hallazgo='NDVI, EVI, LAI, FAPAR and SAVI share r 0.85–0.98 in Sentinel‑2; the four Planet indices share r 0.82–0.97.',
         decision=f"Feeding all {R['n_vars_total']} variables overfit and no model beat the municipality mean. A compact set of {R['n_vars'] - 1} variables plus neighbors' yield did."),
]
nota = (f"Lower RMSE is better. The final model lowers RMSE from {base['rmse']:.3f} (baseline) to {m_best['rmse']:.3f} t/ha, "
        f"but its MAE ({m_best['mae']:.3f}) is slightly higher than the baseline's ({base['mae']:.3f}): it cuts the large errors at the cost of more small ones. "
        f"Using all {R['n_vars_total']} variables at once does worse because of overfitting.")
metodo = [
    dict(paso='Step 1', titulo='Data integration', items=[
        'Sentinel‑2, Landsat (VI6T) and Planet are kept apart by sensor, as the index dictionary recommends.',
        'Scenes with more than 20 % cloud cover are dropped.',
        'CHIRPS/CHIRTS climate (~5 km) is interpolated bilinearly to each parcel centroid.',
        'Elevation and slope (120 m) are averaged over the pixels inside each polygon, reprojected to EPSG:6372.']),
    dict(paso='Step 2', titulo='Variables per parcel', items=[
        f"{R['n_vars_total']} variables were built; the final model uses {R['n_vars']}.",
        'Monthly Apr–Nov 2025 means of the 22 Sentinel‑2 indices, Landsat VI6T and the 4 Planet indices.',
        'NDVI/LAI 2025 phenology: peak, peak date, green‑up, end of season, area under the curve.',
        '2022–24 vigor history per parcel and 2025 rain and temperature anomalies.',
        'Area, location, state, municipality and record format.']),
    dict(paso='Step 3', titulo='Model & validation', items=[
        f"{len(met) - 2} models and two references were compared on the same split: 5‑fold cross‑validation repeated 5 times on the 138 parcels.",
        f"With all {R['n_vars_total']} variables no model beat the municipality mean; {R['n_vars'] - 1} agronomic variables plus the distance‑weighted yield of the 5 nearest training parcels did.",
        f"Final model: Extra Trees (500 trees, min. leaf 3), out‑of‑sample RMSE {m_best['rmse']:.2f} t/ha and R² {m_best['r2']:.2f}.",
        'Forecasts are capped to the training range (2.0–5.59 t/ha).']),
]
src_mf = open('/home/claude/proy/src/modelo_final.py').read()
COMPACT = eval(re.search(r"COMPACT = (\[.*?\])", src_mf, re.S).group(1)) + ['rend_vecinos']
GROUPS = [('Location & management', lambda v: v in ('rend_vecinos', 'lon', 'lat', 'log_area', 'cultivo_mayusculas')),
          ('Topography', lambda v: v.startswith(('elev', 'pend'))),
          ('Climate 2025', lambda v: v.startswith(('prec', 'tmax', 'tmin', 'amp', 'gdd'))),
          ('Sentinel‑2, 2025', lambda v: v.startswith('s2_')),
          ('NDVI history 2022–24', lambda v: v.startswith('hist') or v == 'ndvi_max_vs_hist'),
          ('Planet & Landsat, 2025', lambda v: v.startswith(('pl_', 'ls_')))]
vars_ = [dict(grupo=g, items=[label(v) for v in COMPACT if f(v)]) for g, f in GROUPS]
assert sum(len(g['items']) for g in vars_) == len(COMPACT), 'variable sin grupo'
data = dict(parcelas=parcelas, geom=json.load(open(O + 'geom.json')), ndvi=json.load(open(O + 'ndvi.json')), metricas=met,
            elegido=FINAL, base=BASE, componentes=[FINAL], importancia=R['importancia'], metodo=metodo, nota=nota, eda=eda, vars=vars_)
html = open('/home/claude/proy/web/plantilla.html').read().replace('__DATA__', json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/'))
open('/home/claude/proy/web/rendimiento_agrocebada_2025.html', 'w').write(html)
print(len(html) // 1024, 'KB'); print(json.dumps(eda, ensure_ascii=False, indent=1)); print(nota)
