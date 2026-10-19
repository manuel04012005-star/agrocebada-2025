"""Experimentos rápidos: ¿qué agrega valor sobre la media del municipio?"""
import sys, numpy as np, pandas as pd, warnings
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
sys.path.insert(0, '/home/claude/proy/src'); import shp
warnings.filterwarnings('ignore'); SEED = 2026

F = pd.read_csv('/home/claude/proy/out/features.csv', index_col=0)
g = shp.read_dbf().set_index('ID_POLIGONO')
F['cultivo_mayusculas'] = (g.loc[F.index, 'Cultivo'] == 'CEBADA').astype(int)
tr = F.CONJUNTO == 'ENTRENAMIENTO'; T = F[tr].copy(); y = T.RENDIMIENTO_T_HA.values
num = T.drop(columns=['CONJUNTO', 'RENDIMIENTO_T_HA', 'AREA_HA', 'Estado', 'Municipio'])
# coordenadas métricas aproximadas (km)
xy = np.c_[(F.lon - F.lon.mean()) * 104.6, (F.lat - F.lat.mean()) * 110.6]
XY = xy[tr.values]
COMPACT = ['lon', 'lat', 'log_area', 'cultivo_mayusculas', 'elev_media', 'pend_media', 'prec_ciclo_2025', 'prec_2025_05', 'prec_2025_04', 'prec_anom_2025',
           'tmax_anom_2025', 'amp_term_jun_sep_2025', 's2_ndvi25_max', 's2_ndvi25_auc', 's2_ndvi25_doy_max', 's2_ndvi25_doy_inicio', 's2_ndvi25_duracion',
           's2_lai25_max', 's2_fapar25_auc', 's2_ndwi25_max', 'hist_ndvi_max', 'hist_ndvi_auc', 'ndvi_max_vs_hist', 'pl_ndvi25_max', 'pl_ndvi25_auc',
           's2_ndvi_m05', 's2_ndvi_m06', 's2_ndvi_m08', 's2_ndvi_m10', 'ls_vi6t_m08']

def knn_feat(XYa, ya, XYb, k, excl_self=False):
    d = np.sqrt(((XYb[:, None, :] - XYa[None, :, :]) ** 2).sum(-1))
    if excl_self: np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1)[:, :k]; w = 1 / (np.take_along_axis(d, idx, 1) + 0.3)
    return (ya[idx] * w).sum(1) / w.sum(1), np.take_along_axis(d, idx, 1).mean(1)

def run(fn, reps=3):
    rk = RepeatedKFold(n_splits=5, n_repeats=reps, random_state=SEED); out = np.zeros((reps, len(y)))
    for i, (a, b) in enumerate(rk.split(T)): out[i // 5, b] = fn(a, b)
    r = np.sqrt(((out - y) ** 2).mean(1)); return r.mean(), r.std(), out.mean(0)

def base_mun(a, b):
    ta = T.iloc[a]; mm = ta.groupby('Municipio').RENDIMIENTO_T_HA.mean(); me = ta.groupby('Estado').RENDIMIENTO_T_HA.mean()
    return T.iloc[b].Municipio.map(mm).fillna(T.iloc[b].Estado.map(me)).fillna(y[a].mean()).values
def base_mun_fmt(a, b):
    ta = T.iloc[a].assign(c=num.cultivo_mayusculas.iloc[a].values); tb = T.iloc[b].assign(c=num.cultivo_mayusculas.iloc[b].values)
    key = lambda t: t.Municipio + '|' + t.c.astype(str)
    mm = ta.groupby(key(ta)).RENDIMIENTO_T_HA.mean()
    return key(tb).map(mm).fillna(pd.Series(base_mun(a, b), index=tb.index)).values
def knn(k):
    return lambda a, b: knn_feat(XY[a], y[a], XY[b], k)[0]

def with_knn(cols, k=5, model='et'):
    def f(a, b):
        ka, _ = knn_feat(XY[a], y[a], XY[a], k, excl_self=True)   # dentro del pliegue, sin la propia parcela
        kb, _ = knn_feat(XY[a], y[a], XY[b], k)
        ma = pd.Series(base_mun(a, a), index=a)  # media municipio (incluye propia; leve sesgo)
        Xa = num.iloc[a][cols].assign(knn=ka); Xb = num.iloc[b][cols].assign(knn=kb)
        m = make_model(model); m.fit(Xa, y[a]); return m.predict(Xb)
    return f
def make_model(kind):
    if kind == 'et': return make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(300, min_samples_leaf=3, max_features=0.5, n_jobs=2, random_state=SEED))
    if kind == 'rf': return make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(300, min_samples_leaf=3, max_features=0.5, n_jobs=2, random_state=SEED))
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30)))
def resid(cols, k=5, kind='et'):
    def f(a, b):
        ka, _ = knn_feat(XY[a], y[a], XY[a], k, excl_self=True); kb, _ = knn_feat(XY[a], y[a], XY[b], k)
        m = make_model(kind); m.fit(num.iloc[a][cols], y[a] - ka); return kb + m.predict(num.iloc[b][cols])
    return f
def blend(f1, f2, w):
    return lambda a, b: w * f1(a, b) + (1 - w) * f2(a, b)

exps = {
    'media municipio': base_mun,
    'media municipio×formato': base_mun_fmt,
    'kNN espacial k=3': knn(3), 'kNN espacial k=5': knn(5), 'kNN espacial k=8': knn(8),
    'ET compacto': lambda a, b: make_model('et').fit(num.iloc[a][COMPACT], y[a]).predict(num.iloc[b][COMPACT]),
    'ET compacto + kNN5': with_knn(COMPACT, 5),
    'RF compacto + kNN5': with_knn(COMPACT, 5, 'rf'),
    'ET todas + kNN5': with_knn(list(num.columns), 5),
    'kNN5 + residuo ET compacto': resid(COMPACT, 5),
    'kNN5 + residuo Ridge compacto': resid(COMPACT, 5, 'ridge'),
    '½ kNN5 + ½ ET compacto+kNN5': blend(knn(5), with_knn(COMPACT, 5), .5),
}
res = {}
for k, f in exps.items():
    m, s, _ = run(f); res[k] = (m, s); print(f'{k:34s} RMSE {m:.3f} ± {s:.3f}', flush=True)
