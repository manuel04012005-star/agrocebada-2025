"""Construye la tabla de variables por parcela (197 filas) para el Reto AgroCebada 2026."""
import sys, glob, numpy as np, pandas as pd
from matplotlib.path import Path
sys.path.insert(0, '/home/claude/proy/src')
import shp, geotif

UP = '/root/.claude/uploads/81985831-d14b-58e2-bb98-e4949aa24c45/'
BASICO = '/mnt/user-data/uploads/Conjunto_datos_BASICO_AgroCebada2026.csv'
PRO = UP + 'b5826c68-Conjunto_datos_PRO_AgroCebada.csv'
YLD = UP + 'cd6171d4-ID_area_rendimiento_70_30_Reto_AgroCebada.csv'
DAT = '/home/claude/datos/'
NUB_MAX = 20          # % máximo de nubosidad aceptado
MESES = range(4, 12)  # abril..noviembre 2025

# ---------------- parcelas y rendimiento ----------------
g = shp.load()
y = pd.read_csv(YLD).iloc[:, :4]
base = g.drop(columns=['rings', 'area_ha', 'CONJUNTO']).merge(y, on='ID_POLIGONO')
base['log_area'] = np.log(base.AREA_HA)
F = base[['ID_POLIGONO', 'Estado', 'Municipio', 'CONJUNTO', 'AREA_HA', 'RENDIMIENTO_T_HA', 'lon', 'lat', 'log_area']].set_index('ID_POLIGONO')

# ---------------- topografía (pixeles dentro del polígono) ----------------
topo = {}
for var, fn in [('elev', 'Elevacion_INEGI_CEM4_120m.tif'), ('pend', 'Pendiente_INEGI_CEM4_120m_grados.tif')]:
    a, (x0, dx, y0, dy) = geotif.read(glob.glob(DAT + '**/' + fn, recursive=True)[0])
    vals_m, vals_s = [], []
    for rings in g.rings:
        xs, ys = geotif.lcc6372(np.concatenate(rings)[:, 0], np.concatenate(rings)[:, 1])
        c0, c1 = int((xs.min() - x0) // dx), int((xs.max() - x0) // dx) + 1
        r0, r1 = int((y0 - ys.max()) // dy), int((y0 - ys.min()) // dy) + 1
        cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
        px, py = x0 + (cc + .5) * dx, y0 - (rr + .5) * dy
        inside = np.zeros(cc.shape, bool)
        for ring in rings:
            rx, ry = geotif.lcc6372(ring[:, 0], ring[:, 1])
            inside |= Path(np.c_[rx, ry]).contains_points(np.c_[px.ravel(), py.ravel()]).reshape(cc.shape)
        v = a[rr[inside], cc[inside]] if inside.any() else np.array([])
        if v.size == 0 or np.all(np.isnan(v)):   # parcela menor a un pixel: valor en el centroide
            v = geotif.sample(a, (x0, dx, y0, dy), [xs.mean()], [ys.mean()])
        vals_m.append(np.nanmean(v)); vals_s.append(np.nanstd(v))
    topo[var + '_media'] = vals_m; topo[var + '_std'] = vals_s
for k, v in topo.items(): F[k] = v

# ---------------- clima (interpolación bilineal al centroide) ----------------
def bilinear(a, gt, lon, lat):
    x0, dx, y0, dy = gt
    fx = (lon - x0) / dx - .5; fy = (y0 - lat) / dy - .5
    c, r = np.floor(fx).astype(int), np.floor(fy).astype(int); wx, wy = fx - c, fy - r
    return (a[r, c] * (1 - wx) * (1 - wy) + a[r, c + 1] * wx * (1 - wy) + a[r + 1, c] * (1 - wx) * wy + a[r + 1, c + 1] * wx * wy)
clim = {}
for var, pat in [('prec', 'PREC_{y}_{m:02d}'), ('tmax', 'Tmax_{y}_{m:02d}'), ('tmin', 'Tmin_{y}_{m:02d}')]:
    for yr in range(2022, 2026):
        for m in range(1, 13):
            a, gt = geotif.read(glob.glob(DAT + '**/' + pat.format(y=yr, m=m) + '.tif', recursive=True)[0])
            clim[(var, yr, m)] = bilinear(a, gt, F.lon.values, F.lat.values)
ciclo = range(4, 11)   # abril..octubre
for yr in range(2022, 2026):
    pass
F['prec_ciclo_2025'] = sum(clim[('prec', 2025, m)] for m in ciclo)
F['prec_jun_ago_2025'] = sum(clim[('prec', 2025, m)] for m in (6, 7, 8))
F['prec_sep_oct_2025'] = sum(clim[('prec', 2025, m)] for m in (9, 10))
for m in ciclo: F[f'prec_2025_{m:02d}'] = clim[('prec', 2025, m)]
F['prec_ciclo_hist'] = np.mean([sum(clim[('prec', yr, m)] for m in ciclo) for yr in (2022, 2023, 2024)], axis=0)
F['prec_anom_2025'] = F.prec_ciclo_2025 - F.prec_ciclo_hist
F['tmax_jun_sep_2025'] = np.mean([clim[('tmax', 2025, m)] for m in (6, 7, 8, 9)], axis=0)
F['tmin_jun_sep_2025'] = np.mean([clim[('tmin', 2025, m)] for m in (6, 7, 8, 9)], axis=0)
F['tmed_jun_sep_2025'] = (F.tmax_jun_sep_2025 + F.tmin_jun_sep_2025) / 2
F['amp_term_jun_sep_2025'] = F.tmax_jun_sep_2025 - F.tmin_jun_sep_2025
dias = {4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31}
F['gdd_ciclo_2025'] = sum(np.clip((clim[('tmax', 2025, m)] + clim[('tmin', 2025, m)]) / 2 - 0, 0, None) * dias[m] for m in ciclo)
F['tmax_anom_2025'] = F.tmax_jun_sep_2025 - np.mean([np.mean([clim[('tmax', yr, m)] for m in (6, 7, 8, 9)], axis=0) for yr in (2022, 2023, 2024)], axis=0)

# ---------------- Sentinel-2 / Landsat (dataset Básico) ----------------
b = pd.read_csv(BASICO)
b['f'] = pd.to_datetime(b.fecha_captura, dayfirst=True)
b = b[b.porcentaje_nubosidad <= NUB_MAX]
s2 = b[(b.sensor == 'Sentinel-2') & b.ndvi_promedio.notna()].copy()
ls = b[(b.sensor == 'Landsat') & b.vi6t_promedio.notna()].copy()
IDX = [c[:-9] for c in b.columns if c.endswith('_promedio') and c != 'vi6t_promedio']

def monthly(df, cols, prefix, year=2025, months=MESES):
    d = df[(df.f.dt.year == year) & df.f.dt.month.isin(list(months))]
    p = d.groupby(['ID_POLIGONO', d.f.dt.month])[cols].mean().unstack(1)
    p.columns = [f'{prefix}_{c.replace("_promedio","")}_m{m:02d}' for c, m in p.columns]
    return p

F = F.join(monthly(s2, [i + '_promedio' for i in IDX] + ['ndvi_std'], 's2'))
F = F.join(monthly(ls, ['vi6t_promedio'], 'ls'))

def fenologia(df, col, prefix, year, d0, d1):
    """Curva diaria interpolada y suavizada (ventana 15 d) → métricas fenológicas."""
    out = {}
    t0 = pd.Timestamp(f'{year}-{d0}'); t1 = pd.Timestamp(f'{year}-{d1}')
    days = pd.date_range(t0, t1, freq='D')
    for pid, d in df[(df.f >= t0 - pd.Timedelta(days=30)) & (df.f <= t1 + pd.Timedelta(days=30))].groupby('ID_POLIGONO'):
        s = d.groupby('f')[col].median().sort_index()
        if len(s) < 5: continue
        s = s.reindex(s.index.union(days)).interpolate('time').ffill().bfill().reindex(days)
        s = s.rolling(15, center=True, min_periods=1).median()
        v = s.values; base_v = np.nanpercentile(v, 10); mx = np.nanmax(v); amp = mx - base_v
        imax = int(np.nanargmax(v)); thr = base_v + .5 * amp
        up = np.where(v[:imax + 1] >= thr)[0]; dn = np.where(v[imax:] < thr)[0]
        sos = up[0] if len(up) else np.nan; eos = imax + dn[0] if len(dn) else len(v) - 1
        out[pid] = {f'{prefix}_max': mx, f'{prefix}_base': base_v, f'{prefix}_amp': amp,
                    f'{prefix}_doy_max': days[imax].dayofyear, f'{prefix}_doy_inicio': days[0].dayofyear + sos,
                    f'{prefix}_doy_fin': days[0].dayofyear + eos, f'{prefix}_duracion': eos - sos,
                    f'{prefix}_auc': np.nansum(np.clip(v - base_v, 0, None)), f'{prefix}_auc_total': np.nansum(v),
                    f'{prefix}_tasa_verdeo': amp / max(imax - sos, 1) if not np.isnan(sos) else np.nan}
    return pd.DataFrame(out).T

F = F.join(fenologia(s2, 'ndvi_promedio', 's2_ndvi25', 2025, '04-01', '12-15'))
F = F.join(fenologia(s2, 'lai_promedio', 's2_lai25', 2025, '04-01', '12-15')[['s2_lai25_max', 's2_lai25_auc']])
F = F.join(fenologia(s2, 'fapar_promedio', 's2_fapar25', 2025, '04-01', '12-15')[['s2_fapar25_max', 's2_fapar25_auc']])
F = F.join(fenologia(s2, 'ndwi_promedio', 's2_ndwi25', 2025, '04-01', '12-15')[['s2_ndwi25_max', 's2_ndwi25_auc']])
# historial 2022-2024 (vigor promedio de la parcela en años previos)
hist = [fenologia(s2, 'ndvi_promedio', f'h', yr, '04-01', '12-15')[['h_max', 'h_auc', 'h_doy_max']] for yr in (2022, 2023, 2024)]
H = sum(hh for hh in hist) / 3
H.columns = ['hist_ndvi_max', 'hist_ndvi_auc', 'hist_doy_max']
F = F.join(H)
F['ndvi_max_vs_hist'] = F.s2_ndvi25_max - F.hist_ndvi_max

# ---------------- Planet (dataset PRO, 2025) ----------------
p = pd.read_csv(PRO)
p['f'] = pd.to_datetime(p.fecha_captura, dayfirst=True)
p = p[(p.porcentaje_nubosidad <= NUB_MAX) & p.ndvi_promedio.notna()]
F = F.join(monthly(p, ['ndvi_promedio', 'evi_promedio', 'lai_promedio', 'msavi_promedio'], 'pl', months=range(4, 12)))
F = F.join(fenologia(p, 'ndvi_promedio', 'pl_ndvi25', 2025, '04-01', '12-15')[['pl_ndvi25_max', 'pl_ndvi25_auc', 'pl_ndvi25_doy_max']])

F.index.name = 'ID_POLIGONO'
F.to_csv('/home/claude/proy/out/features.csv')
print(F.shape); print(F.isna().sum()[lambda s: s > 0].sort_values().tail(15))
print(F[['elev_media', 'pend_media', 'prec_ciclo_2025', 'tmed_jun_sep_2025', 's2_ndvi25_max', 's2_ndvi25_doy_max', 'pl_ndvi25_max']].describe().round(2).T)
