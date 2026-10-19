"""Entrena, valida (5-fold x 5 repeticiones) y predice el rendimiento de cebada."""
import sys, json, numpy as np, pandas as pd, warnings
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
sys.path.insert(0, '/home/claude/proy/src'); import shp
warnings.filterwarnings('ignore')
SEED = 2026

F = pd.read_csv('/home/claude/proy/out/features.csv', index_col=0)
g = shp.read_dbf().set_index('ID_POLIGONO')
F['cultivo_mayusculas'] = (g.loc[F.index, 'Cultivo'] == 'CEBADA').astype(int)   # formato del registro de origen
X = pd.get_dummies(F.drop(columns=['CONJUNTO', 'RENDIMIENTO_T_HA', 'AREA_HA']), columns=['Estado', 'Municipio'], dtype=float)
tr = F.CONJUNTO == 'ENTRENAMIENTO'
Xtr, ytr, Xte = X[tr], F.loc[tr, 'RENDIMIENTO_T_HA'].values, X[~tr]

modelos = {
    'Ridge': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 40))),
    'ElasticNet': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), ElasticNetCV(l1_ratio=0.5, alphas=20, cv=4, max_iter=4000, tol=1e-3, random_state=SEED)),
    'Random Forest': make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(500, min_samples_leaf=2, max_features=0.3, n_jobs=-1, random_state=SEED)),
    'Extra Trees': make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(500, min_samples_leaf=2, max_features=0.3, n_jobs=-1, random_state=SEED)),
    'Gradient Boosting': make_pipeline(SimpleImputer(strategy='median'), GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, min_samples_leaf=4, random_state=SEED)),
    'HistGB': HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=8, l2_regularization=1.0, random_state=SEED),
}

rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED)
splits = list(rkf.split(Xtr))
oof = {k: np.zeros((5, len(ytr))) for k in list(modelos) + ['Línea base (media municipio)']}
for i, (a, b) in enumerate(splits):
    rep = i // 5
    for k, m in modelos.items():
        m.fit(Xtr.iloc[a], ytr[a]); oof[k][rep, b] = m.predict(Xtr.iloc[b])
    # línea base: media del municipio (o del estado si no hay datos)
    tra = F[tr].iloc[a]; tsb = F[tr].iloc[b]
    mm = tra.groupby('Municipio').RENDIMIENTO_T_HA.mean(); me = tra.groupby('Estado').RENDIMIENTO_T_HA.mean()
    oof['Línea base (media municipio)'][rep, b] = tsb.Municipio.map(mm).fillna(tsb.Estado.map(me)).fillna(tra.RENDIMIENTO_T_HA.mean()).values

def met(p):
    r = [(np.sqrt(mean_squared_error(ytr, q)), mean_absolute_error(ytr, q), r2_score(ytr, q)) for q in p]
    r = np.array(r); return dict(rmse=r[:, 0].mean(), rmse_sd=r[:, 0].std(), mae=r[:, 1].mean(), r2=r[:, 2].mean())
res = {k: met(v) for k, v in oof.items()}
orden = sorted(modelos, key=lambda k: res[k]['rmse'])
top = orden[:3]
oof['Ensamble'] = np.mean([oof[k] for k in top], axis=0); res['Ensamble'] = met(oof['Ensamble'])
R = pd.DataFrame(res).T.sort_values('rmse'); print(R.round(3)); print('ensamble de:', top)

if '--cv-only' in sys.argv:
    json.dump({'metricas': R.reset_index().rename(columns={'index': 'modelo'}).to_dict('records'), 'n_vars': X.shape[1]}, open('/home/claude/proy/out/resultados.json', 'w'), ensure_ascii=False, indent=1)
    sys.exit(0)
# ---------------- modelo final ----------------
mejor = 'Ensamble' if res['Ensamble']['rmse'] <= res[orden[0]]['rmse'] else orden[0]
usar = top if mejor == 'Ensamble' else [orden[0]]
preds, imps = [], []
for k in usar:
    m = modelos[k]; m.fit(Xtr, ytr); preds.append(m.predict(Xte))
    pi = permutation_importance(m, Xtr, ytr, n_repeats=5, random_state=SEED, scoring='neg_root_mean_squared_error')
    imps.append(pd.Series(pi.importances_mean, index=X.columns))
pred = np.clip(np.mean(preds, axis=0), ytr.min(), ytr.max())
spread = np.std(preds, axis=0) if len(preds) > 1 else np.zeros(len(pred))
imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
print('importancia:\n', imp.head(15).round(4))

# intervalo empírico 80% a partir de residuos fuera de muestra del modelo elegido
oof_best = oof[mejor].mean(axis=0); resid = ytr - oof_best
q10, q90 = np.quantile(resid, [.1, .9])
out = pd.DataFrame({'ID_POLIGONO': Xte.index, 'RENDIMIENTO_PRED_T_HA': pred.round(2),
                    'LIM_INF_80': np.clip(pred + q10, 0, None).round(2), 'LIM_SUP_80': (pred + q90).round(2)})
out.to_csv('/home/claude/proy/out/predicciones_59_parcelas.csv', index=False)
print(out.describe().round(2))

json.dump({'metricas': R.reset_index().rename(columns={'index': 'modelo'}).to_dict('records'), 'elegido': mejor, 'componentes': usar,
           'oof': pd.DataFrame({'ID_POLIGONO': Xtr.index, 'real': ytr, 'pred': oof_best.round(3)}).to_dict('records'),
           'importancia': [{'var': k, 'imp': round(float(v), 4)} for k, v in imp.head(15).items()],
           'pi80': [float(q10), float(q90)], 'n_vars': X.shape[1]},
          open('/home/claude/proy/out/resultados.json', 'w'), ensure_ascii=False, indent=1)
