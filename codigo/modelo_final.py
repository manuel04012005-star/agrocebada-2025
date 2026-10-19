"""Modelo final: Extra Trees con 30 variables agronómicas + rendimiento de vecinos (kNN espacial).
Misma partición de validación (RepeatedKFold 5x5, semilla 2026) que la comparación con 265 variables (modelo.py)."""
import sys, json, numpy as np, pandas as pd, warnings
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
sys.path.insert(0, '/home/claude/proy/src'); import shp
warnings.filterwarnings('ignore'); SEED = 2026; K = 5

F = pd.read_csv('/home/claude/proy/out/features.csv', index_col=0)
g = shp.read_dbf().set_index('ID_POLIGONO')
F['cultivo_mayusculas'] = (g.loc[F.index, 'Cultivo'] == 'CEBADA').astype(int)
tr = (F.CONJUNTO == 'ENTRENAMIENTO').values; T = F[tr]; y = T.RENDIMIENTO_T_HA.values
XYall = np.c_[(F.lon - F.lon.mean()) * 104.6, (F.lat - F.lat.mean()) * 110.6]   # km aprox.
XY, XYp = XYall[tr], XYall[~tr]
COMPACT = ['lon', 'lat', 'log_area', 'cultivo_mayusculas', 'elev_media', 'pend_media', 'prec_ciclo_2025', 'prec_2025_05', 'prec_2025_04', 'prec_anom_2025',
           'tmax_anom_2025', 'amp_term_jun_sep_2025', 's2_ndvi25_max', 's2_ndvi25_auc', 's2_ndvi25_doy_max', 's2_ndvi25_doy_inicio', 's2_ndvi25_duracion',
           's2_lai25_max', 's2_fapar25_auc', 's2_ndwi25_max', 'hist_ndvi_max', 'hist_ndvi_auc', 'ndvi_max_vs_hist', 'pl_ndvi25_max', 'pl_ndvi25_auc',
           's2_ndvi_m05', 's2_ndvi_m06', 's2_ndvi_m08', 's2_ndvi_m10', 'ls_vi6t_m08']

def knn(XYa, ya, XYb, k=K, self_excl=False):
    d = np.sqrt(((XYb[:, None] - XYa[None]) ** 2).sum(-1))
    if self_excl: np.fill_diagonal(d, np.inf)
    i = np.argsort(d, 1)[:, :k]; w = 1 / (np.take_along_axis(d, i, 1) + 0.3)
    return (ya[i] * w).sum(1) / w.sum(1)
def design(a_idx, b_XY, b_rows):
    """Matriz con la variable kNN calculada solo con parcelas de entrenamiento del pliegue."""
    Xa = T.iloc[a_idx][COMPACT].assign(rend_vecinos=knn(XY[a_idx], y[a_idx], XY[a_idx], self_excl=True))
    Xb = b_rows[COMPACT].assign(rend_vecinos=knn(XY[a_idx], y[a_idx], b_XY))
    return Xa, Xb
MODELS = {
    'Extra Trees (30 var. + vecinos)': lambda: make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(500, min_samples_leaf=3, max_features=0.5, n_jobs=2, random_state=SEED)),
    'Random Forest (30 var. + vecinos)': lambda: make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(500, min_samples_leaf=3, max_features=0.5, n_jobs=2, random_state=SEED)),
    'Ridge (30 var. + vecinos)': lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30))),
}
FINAL = 'Extra Trees (30 var. + vecinos)'
splits = list(RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(T))
oof = {k: np.zeros((5, len(y))) for k in list(MODELS) + ['Línea base (media municipio)', 'Vecinos espaciales (kNN, k=5)']}
for s, (a, b) in enumerate(splits):
    r = s // 5
    for k, mk in MODELS.items():
        Xa, Xb = design(a, XY[b], T.iloc[b]); oof[k][r, b] = mk().fit(Xa, y[a]).predict(Xb)
    ta, tb = T.iloc[a], T.iloc[b]
    mm = ta.groupby('Municipio').RENDIMIENTO_T_HA.mean(); me = ta.groupby('Estado').RENDIMIENTO_T_HA.mean()
    oof['Línea base (media municipio)'][r, b] = tb.Municipio.map(mm).fillna(tb.Estado.map(me)).fillna(y[a].mean()).values
    oof['Vecinos espaciales (kNN, k=5)'][r, b] = knn(XY[a], y[a], XY[b])
def met(p):
    m = np.array([(np.sqrt(((q - y) ** 2).mean()), mean_absolute_error(y, q), r2_score(y, q)) for q in p])
    return dict(rmse=m[:, 0].mean(), rmse_sd=m[:, 0].std(), mae=m[:, 1].mean(), r2=m[:, 2].mean())
res = [dict(modelo=k, grupo='30 variables + vecinos' if '30 var' in k else 'Referencia', **met(v)) for k, v in oof.items()]
OLD = json.load(open('/home/claude/proy/out/resultados.json')); old = OLD['metricas']; NT = OLD['n_vars']   # misma partición, todas las variables
res += [dict(m, modelo=m['modelo'] + f' ({NT} var.)', grupo=f'{NT} variables') for m in old if not m['modelo'].startswith('Línea') and m['modelo'] != 'Ensamble']
res = sorted(res, key=lambda m: m['rmse'])
print(pd.DataFrame(res).set_index('modelo').round(3))

# ---------- ajuste final con las 138 parcelas ----------
P = F[~tr]
Xa, Xb = design(np.arange(len(y)), XYp, P)
m = MODELS[FINAL]().fit(Xa, y)
pred = np.clip(m.predict(Xb), y.min(), y.max())
o = oof[FINAL].mean(0); q10, q90 = np.quantile(y - o, [.1, .9])
cov = np.mean([((y - oo) >= q10) & ((y - oo) <= q90) for oo in oof[FINAL]])
pi = permutation_importance(m, Xa, y, n_repeats=20, random_state=SEED, scoring='neg_root_mean_squared_error')
imp = pd.Series(pi.importances_mean, index=Xa.columns).sort_values(ascending=False)
print(imp.head(12).round(4)); print('cobertura PI80 oof', round(cov, 3), 'q', round(q10, 3), round(q90, 3))

out = pd.DataFrame({'ID_POLIGONO': P.index, 'RENDIMIENTO_PRED_T_HA': pred.round(2),
                    'LIM_INF_80': np.clip(pred + q10, 0, None).round(2), 'LIM_SUP_80': (pred + q90).round(2)})
out.to_csv('/home/claude/proy/out/predicciones_59_parcelas.csv', index=False)
out[['ID_POLIGONO', 'RENDIMIENTO_PRED_T_HA']].rename(columns={'RENDIMIENTO_PRED_T_HA': 'RENDIMIENTO_T_HA'}).to_csv('/home/claude/proy/out/entrega_59_parcelas.csv', index=False)
print(out.describe().round(2))
json.dump({'metricas': res, 'elegido': FINAL, 'componentes': [FINAL], 'n_vars': len(COMPACT) + 1, 'n_vars_total': NT,
           'oof': pd.DataFrame({'ID_POLIGONO': T.index, 'real': y, 'pred': o.round(3)}).to_dict('records'),
           'importancia': [{'var': k, 'imp': round(float(v), 4)} for k, v in imp.head(12).items()], 'pi80': [float(q10), float(q90)]},
          open('/home/claude/proy/out/resultados_final.json', 'w'), ensure_ascii=False, indent=1)
