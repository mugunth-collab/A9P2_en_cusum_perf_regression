#!/usr/bin/env python3
"""A9P2 - Continuous performance profiling in mobile CI/CD for early detection of performance regressions
(smart-grid consumer energy-monitoring app).

Proposed: elastic-net logistic regression on stream-normalised profiling evidence + one-sided CUSUM
change-point statistics, with a two-key CUSUM gate on the alarm decision.
Baselines: fixed per-stream performance budgets, percent-delta rule vs parent commit, random forest.

Run:  python3 run_all.py      (reads ../data, writes ../results and ../figures; about 2-4 min on 1 core)
"""
import os, json, time, warnings, shutil, subprocess
warnings.filterwarnings('ignore')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score, precision_score,
                             recall_score, precision_recall_curve)

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'A9P2_energyapp_perf_profiling.csv')
RES = os.path.join(HERE, '..', 'results'); FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
SEED = 2026
np.random.seed(SEED)

plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9.5, 'axes.titlesize': 9.5, 'axes.labelsize': 9.5,
                     'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8, 'figure.dpi': 100,
                     'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5})
C_PROP, C_RF, C_DELTA, C_FIX = '#c0392b', '#2c7fb8', '#7f7f7f', '#e69f00'

def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(RES, name + '.jpg'), dpi=300, bbox_inches='tight', pil_kwargs={'quality': 92})
    plt.close(fig)

# ------------------------------------------------------------------ 1. load + pre-processing (causal, per stream)
d = pd.read_csv(DATA, parse_dates=['build_ts']).sort_values('build_ts', kind='mergesort').reset_index(drop=True)
y = d['perf_regression'].values.astype(int)
d['stream'] = d['device_class'] + '|' + d['benchmark_scenario']
W, MINP = 40, 8          # rolling window of previous runs in the same device x scenario stream
miss = d[['app_bundle_mb', 'frame_time_p90_ms', 'energy_mah']].isna().mean().to_dict()

def roll_prev_median(s):
    return s.shift(1).rolling(W, min_periods=1).median()

for c in ['app_bundle_mb', 'frame_time_p90_ms', 'energy_mah']:   # causal imputation: median of previous runs in stream
    fill = d.groupby('stream')[c].transform(roll_prev_median)
    d[c] = d[c].fillna(fill).fillna(d[c].median())

d['r_lat'] = np.log(d['p50_latency_ms'] / d['baseline_p50_latency_ms'])         # log latency ratio vs parent commit
d['m_frame'] = np.log(d['frame_time_p90_ms'])
d['m_jank'] = np.log1p(d['jank_frames_pct'])
d['m_mem'] = np.log(d['peak_memory_mb'])
d['m_cpu'] = np.log(d['cpu_time_ms'] / d['baseline_p50_latency_ms'])
d['m_energy'] = np.log(d['energy_mah'] + 1e-3)
d['m_bundle'] = np.log(d['app_bundle_mb'])
METRICS = ['r_lat', 'm_frame', 'm_jank', 'm_mem', 'm_cpu', 'm_energy', 'm_bundle']
ZCLIP = 8.0

def robust_z(s):
    med = s.shift(1).rolling(W, min_periods=MINP).median()
    mad = (s - med).abs().shift(1).rolling(W, min_periods=MINP).median() * 1.4826
    return ((s - med) / (mad + 1e-6)).clip(-ZCLIP, ZCLIP).fillna(0.0)

for m in METRICS:
    d['z_' + m] = d.groupby('stream')[m].transform(robust_z)
n_clipped = int((d[['z_' + m for m in METRICS]].abs() >= ZCLIP).sum().sum())

def cusum(z, k, h):
    """one-sided CUSUM with restart after a signal: S_t = max(0, S_{t-1} + z_t - k)"""
    S = np.zeros(len(z)); s = 0.0
    for i, v in enumerate(z):
        s = max(0.0, s + v - k); S[i] = s
        if s > h: s = 0.0
    return S

def add_cusum(df, k, h):
    out = {}
    for src, nm in [('z_r_lat', 'S_lat'), ('z_m_frame', 'S_frame')]:
        out[nm] = df.groupby('stream')[src].transform(lambda z: pd.Series(cusum(z.values, k, h), index=z.index)).values
    return out

K_CUSUM, H_CUSUM = 0.5, 4.0
cs = add_cusum(d, K_CUSUM, H_CUSUM)
d['S_lat'], d['S_frame'] = cs['S_lat'], cs['S_frame']

SCEN = sorted(d['benchmark_scenario'].unique())
CODE_BIN = ['touches_ui_layer', 'touches_network_layer', 'touches_local_db', 'chart_library_upgraded', 'static_main_thread_io_flag']

def code_features(df):
    X = pd.DataFrame(index=df.index)
    X['log_loc_add'] = np.log1p(df['loc_added']); X['log_loc_del'] = np.log1p(df['loc_deleted'])
    X['log_files'] = np.log1p(df['files_changed'])
    for c in CODE_BIN: X[c] = df[c]
    X['new_deps'] = df['new_dependencies']; X['img_assets'] = df['image_assets_added']
    X['log_points'] = np.log(df['meter_interval_points_rendered'])
    X['ios'] = (df['platform'] == 'ios').astype(float)
    for s in SCEN: X['sc_' + s] = (df['benchmark_scenario'] == s).astype(float)
    for c in ['touches_ui_layer', 'touches_local_db', 'chart_library_upgraded', 'static_main_thread_io_flag']:
        for s in SCEN: X[c[:10] + 'x' + s[:10]] = df[c] * (df['benchmark_scenario'] == s)   # code area x user journey
    return X

def runtime_features(df, normalised=True, with_cusum=True):
    X = pd.DataFrame(index=df.index)
    if normalised:
        for m in METRICS: X['z_' + m] = df['z_' + m]
    else:
        for m in METRICS: X[m] = df[m]
    X['r_lat'] = df['r_lat']
    if with_cusum: X['S_lat'] = df['S_lat']; X['S_frame'] = df['S_frame']
    return X

def build_X(df, code=True, runtime=True, normalised=True, with_cusum=True):
    parts = []
    if code: parts.append(code_features(df))
    if runtime: parts.append(runtime_features(df, normalised, with_cusum))
    return pd.concat(parts, axis=1).astype(float)

X_full = build_X(d)
print(f'[prep] {len(d)} runs, {X_full.shape[1]} features, prevalence {y.mean():.3f}, {time.time()-T0:.1f}s')

# ------------------------------------------------------------------ 2. models and decision rules
EN_LAM, EN_L1 = 0.003, 0.5

class ENLogit:
    """Elastic-net logistic regression fitted by FISTA (accelerated proximal gradient), Eq. (6)-(7).
    min_w,b (1/n) sum logloss + lam * (rho * |w|_1 + (1 - rho)/2 * |w|_2^2); intercept b unpenalised."""
    def __init__(self, lam=EN_LAM, rho=EN_L1, max_iter=800, tol=2e-5):
        self.lam, self.rho, self.max_iter, self.tol = lam, rho, max_iter, tol
    def fit(self, X, yy):
        n, p = X.shape; Xa = np.hstack([X, np.ones((n, 1))])
        L = 0.25 * np.linalg.norm(Xa, 2) ** 2 / n + self.lam * (1 - self.rho)   # Lipschitz constant of smooth part
        w = np.zeros(p + 1); w[-1] = np.log(yy.mean() / (1 - yy.mean())); z = w.copy(); tk = 1.0
        l1, l2 = self.lam * self.rho, self.lam * (1 - self.rho)
        for it in range(self.max_iter):
            mu = 1 / (1 + np.exp(-np.clip(Xa @ z, -30, 30)))
            grad = Xa.T @ (mu - yy) / n; grad[:-1] += l2 * z[:-1]
            wn = z - grad / L
            wn[:-1] = np.sign(wn[:-1]) * np.maximum(np.abs(wn[:-1]) - l1 / L, 0)   # soft-threshold (prox of L1)
            if np.dot(z - wn, wn - w) > 0: tk = 1.0                                  # adaptive restart
            tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2; z = wn + (tk - 1) / tn * (wn - w)
            if np.max(np.abs(wn - w)) < self.tol: w = wn; break
            w, tk = wn, tn
        self.coef_, self.intercept_, self.n_iter_ = w[:-1][None, :], w[-1], it + 1
        return self
    def predict_proba(self, X):
        p1 = 1 / (1 + np.exp(-np.clip(X @ self.coef_[0] + self.intercept_, -30, 30)))
        return np.column_stack([1 - p1, p1])

def fit_en(X, yy, lam=EN_LAM, l1=EN_L1):
    sc = StandardScaler().fit(X)
    return sc, ENLogit(lam, l1).fit(sc.transform(X), yy.astype(float))

def pred_en(model, X):
    sc, m = model; return m.predict_proba(sc.transform(X))[:, 1]

def inner_oof(X, yy, lam=EN_LAM, l1=EN_L1, k=3):
    p = np.zeros(len(yy))
    for tr, va in StratifiedKFold(k, shuffle=True, random_state=SEED).split(X, yy):
        p[va] = pred_en(fit_en(X.iloc[tr], yy[tr], lam, l1), X.iloc[va])
    return p

TGRID = np.round(np.arange(0.10, 0.951, 0.025), 3)
GGRID = [0.5, 1.0, 2.0, 3.0]

def tune_single(p, yy):
    f = [f1_score(yy, p >= t) for t in TGRID]; return float(TGRID[int(np.argmax(f))])

def gate(p, S, tlo, thi, g):
    return (p >= thi) | ((p >= tlo) & (S >= g))

def tune_gate(p, S, yy):
    yb = yy.astype(bool); P = yb.sum()
    hi = p[None, :] >= TGRID[:, None]                       # (T, n)
    best = (-1, None)
    for g in GGRID:
        lo = hi & (S >= g)[None, :]
        for i, thi in enumerate(TGRID):
            for j in range(i):
                a = hi[i] | lo[j]; tp = np.sum(a & yb); f = 2 * tp / (a.sum() + P)
                if f > best[0] + 1e-9: best = (f, (float(TGRID[j]), float(thi), g))
    return best[1]

def metrics(yy, alarm, score=None):
    out = {'f1': f1_score(yy, alarm), 'precision': precision_score(yy, alarm, zero_division=0),
           'recall': recall_score(yy, alarm), 'far': float(np.mean(alarm[yy == 0]))}
    if score is not None:
        out['pr_auc'] = average_precision_score(yy, score); out['roc_auc'] = roc_auc_score(yy, score)
    return out

def fixed_budget_rule(dtr, ytr, dte):
    """per-stream absolute budgets on p50 latency and p90 frame time = quantile q of training runs (q tuned on train)"""
    best = (-1, None)
    for q in [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]:
        b = dtr.groupby('stream')[['p50_latency_ms', 'frame_time_p90_ms']].quantile(q)
        bl = dtr['stream'].map(b['p50_latency_ms']); bf = dtr['stream'].map(b['frame_time_p90_ms'])
        f = f1_score(ytr, (dtr['p50_latency_ms'] > bl) | (dtr['frame_time_p90_ms'] > bf))
        if f > best[0]: best = (f, q)
    q = best[1]
    b = dtr.groupby('stream')[['p50_latency_ms', 'frame_time_p90_ms']].quantile(q)
    alarm = ((dte['p50_latency_ms'] > dte['stream'].map(b['p50_latency_ms'])) |
             (dte['frame_time_p90_ms'] > dte['stream'].map(b['frame_time_p90_ms']))).values
    score = np.maximum(dte['p50_latency_ms'] / dte['stream'].map(b['p50_latency_ms']),
                       dte['frame_time_p90_ms'] / dte['stream'].map(b['frame_time_p90_ms'])).values
    return alarm, score, q

DGRID = np.round(np.arange(0.0, 0.81, 0.02), 2)
def delta_rule(dtr, ytr, dte):
    rel_tr = dtr['p50_latency_ms'] / dtr['baseline_p50_latency_ms'] - 1
    f = [f1_score(ytr, rel_tr > dl) for dl in DGRID]; dl = float(DGRID[int(np.argmax(f))])
    rel = (dte['p50_latency_ms'] / dte['baseline_p50_latency_ms'] - 1).values
    return rel > dl, rel, dl

def fit_rf(X, yy):
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, max_features='sqrt', max_samples=0.6, oob_score=True,
                                n_jobs=1, random_state=SEED).fit(X, yy)
    t = tune_single(rf.oob_decision_function_[:, 1], yy)
    return rf, t

# ------------------------------------------------------------------ 3. main 5-fold CV comparison + ablation
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
FOLDS = list(skf.split(X_full, y))
rows, abl_rows = [], []
oof = {k: np.zeros(len(y)) for k in ['prop', 'rf', 'delta', 'fixed']}
oof_alarm = {k: np.zeros(len(y), bool) for k in ['prop', 'rf', 'delta', 'fixed']}
tuned = []

ABL = {  # name: (feature builder kwargs, en kwargs, use_gate)
    'Full (EN + CUSUM feat. + gate)': (dict(), dict(), True),
    'No CUSUM gate': (dict(), dict(), False),
    'No CUSUM (feat. or gate)': (dict(with_cusum=False), dict(), False),
    'No stream normalisation': (dict(normalised=False), dict(), True),
    'Lasso (l1 ratio 1)': (dict(), dict(l1=1.0), True),
    'Ridge (l1 ratio 0)': (dict(), dict(l1=0.0), True),
    'Runtime metrics only': (dict(code=False), dict(), True),
    'Code features only': (dict(runtime=False), dict(), False),
}
for fi, (tr, te) in enumerate(FOLDS):
    ytr, yte = y[tr], y[te]
    # proposed
    Xtr, Xte = X_full.iloc[tr], X_full.iloc[te]
    pin = inner_oof(Xtr, ytr); tlo, thi, g = tune_gate(pin, d['S_lat'].values[tr], ytr)
    mod = fit_en(Xtr, ytr); p = pred_en(mod, Xte); al = gate(p, d['S_lat'].values[te], tlo, thi, g)
    oof['prop'][te] = p; oof_alarm['prop'][te] = al
    rows.append(dict(fold=fi, method='EN+CUSUM (proposed)', **metrics(yte, al, p)))
    # random forest (same features)
    rf, trf = fit_rf(Xtr, ytr); prf = rf.predict_proba(Xte)[:, 1]; alrf = prf >= trf
    oof['rf'][te] = prf; oof_alarm['rf'][te] = alrf
    rows.append(dict(fold=fi, method='Random forest', **metrics(yte, alrf, prf)))
    # percent-delta
    ald, sd, dl = delta_rule(d.iloc[tr], ytr, d.iloc[te]); oof['delta'][te] = sd; oof_alarm['delta'][te] = ald
    rows.append(dict(fold=fi, method='Percent-delta rule', **metrics(yte, ald, sd)))
    # fixed budgets
    alf, sf, q = fixed_budget_rule(d.iloc[tr], ytr, d.iloc[te]); oof['fixed'][te] = sf; oof_alarm['fixed'][te] = alf
    rows.append(dict(fold=fi, method='Fixed thresholds', **metrics(yte, alf, sf)))
    tuned.append(dict(fold=fi, tau_lo=tlo, tau_hi=thi, gamma=g, rf_tau=trf, delta=dl, fixed_q=q,
                      en_nonzero=int(np.sum(mod[1].coef_ != 0)), n_features=X_full.shape[1]))
    # ablation
    for name, (fk, ek, use_gate) in ABL.items():
        if name == 'Full (EN + CUSUM feat. + gate)':
            abl_rows.append(dict(fold=fi, variant=name, **metrics(yte, al, p))); continue
        Xa = build_X(d, **fk); Xatr, Xate = Xa.iloc[tr], Xa.iloc[te]
        pin_a = inner_oof(Xatr, ytr, **ek); m_a = fit_en(Xatr, ytr, **ek); pa = pred_en(m_a, Xate)
        if use_gate:
            a, b, c = tune_gate(pin_a, d['S_lat'].values[tr], ytr); ala = gate(pa, d['S_lat'].values[te], a, b, c)
        else:
            ala = pa >= tune_single(pin_a, ytr)
        abl_rows.append(dict(fold=fi, variant=name, **metrics(yte, ala, pa)))
    print(f'[cv] fold {fi} done, {time.time()-T0:.1f}s')

cv = pd.DataFrame(rows); cv.to_csv(os.path.join(RES, 'cv_fold_metrics.csv'), index=False)
pd.DataFrame(tuned).to_csv(os.path.join(RES, 'tuned_thresholds_per_fold.csv'), index=False)
ab = pd.DataFrame(abl_rows); ab.to_csv(os.path.join(RES, 'ablation_fold_metrics.csv'), index=False)
MORDER = ['Fixed thresholds', 'Percent-delta rule', 'Random forest', 'EN+CUSUM (proposed)']
cols = ['pr_auc', 'roc_auc', 'f1', 'precision', 'recall', 'far']
main = cv.groupby('method')[cols].agg(['mean', 'std']).loc[MORDER]
main.columns = [f'{a}_{b}' for a, b in main.columns]; main.round(4).to_csv(os.path.join(RES, 'table_main_cv.csv'))
abl = ab.groupby('variant')[['pr_auc', 'f1', 'precision', 'recall', 'far']].agg(['mean', 'std']).loc[list(ABL)]
abl.columns = [f'{a}_{b}' for a, b in abl.columns]; abl.round(4).to_csv(os.path.join(RES, 'table_ablation.csv'))
print(main[['pr_auc_mean', 'f1_mean', 'precision_mean', 'recall_mean', 'far_mean']].round(3))
print(abl[['pr_auc_mean', 'f1_mean']].round(3))

# paired tests on fold metrics (proposed vs each baseline)
tests = {}
for mth in MORDER[:-1]:
    for met in ['f1', 'pr_auc']:
        a = cv[cv.method == 'EN+CUSUM (proposed)'].sort_values('fold')[met].values
        b = cv[cv.method == mth].sort_values('fold')[met].values
        tests[f'{mth}|{met}'] = dict(mean_diff=float(np.mean(a - b)), t_p=float(stats.ttest_rel(a, b).pvalue))
# McNemar on pooled out-of-fold alarms: proposed vs random forest
cp, cr = oof_alarm['prop'] == y, oof_alarm['rf'] == y
b01, b10 = int(np.sum(cp & ~cr)), int(np.sum(~cp & cr))
mcn_p = float(stats.binomtest(b01, b01 + b10, 0.5).pvalue) if b01 + b10 > 0 else 1.0

# ------------------------------------------------------------------ 4. temporal hold-out (train 2025, test 2026-H1)
tr_t = np.where(d['build_ts'] < '2026-01-01')[0]; te_t = np.where(d['build_ts'] >= '2026-01-01')[0]
Xtr, Xte = X_full.iloc[tr_t], X_full.iloc[te_t]; ytr, yte = y[tr_t], y[te_t]
pin = inner_oof(Xtr, ytr); tlo, thi, g = tune_gate(pin, d['S_lat'].values[tr_t], ytr)
mod_t = fit_en(Xtr, ytr); p_t = pred_en(mod_t, Xte); al_t = gate(p_t, d['S_lat'].values[te_t], tlo, thi, g)
rf_t, trf_t = fit_rf(Xtr, ytr); prf_t = rf_t.predict_proba(Xte)[:, 1]
ald_t, sd_t, _ = delta_rule(d.iloc[tr_t], ytr, d.iloc[te_t]); alf_t, sf_t, _ = fixed_budget_rule(d.iloc[tr_t], ytr, d.iloc[te_t])
temporal = {'EN+CUSUM (proposed)': metrics(yte, al_t, p_t), 'Random forest': metrics(yte, prf_t >= trf_t, prf_t),
            'Percent-delta rule': metrics(yte, ald_t, sd_t), 'Fixed thresholds': metrics(yte, alf_t, sf_t)}
pd.DataFrame(temporal).T.loc[MORDER].round(4).to_csv(os.path.join(RES, 'table_temporal.csv'))
print('[temporal]', {k: round(v['f1'], 3) for k, v in temporal.items()})

# ------------------------------------------------------------------ 5. pre-merge profiling budget (code-only risk ranking, temporal)
Xc = build_X(d, runtime=False)
mc = fit_en(Xc.iloc[tr_t], ytr); risk = pred_en(mc, Xc.iloc[te_t])
rfc = RandomForestClassifier(200, min_samples_leaf=3, max_samples=0.6, n_jobs=1, random_state=SEED).fit(Xc.iloc[tr_t], ytr)
risk_rf = rfc.predict_proba(Xc.iloc[te_t])[:, 1]
minutes = d['p50_latency_ms'].values[te_t]           # device-farm time proxy: one profiling session ~ scenario latency x reps
def budget_curve(score):
    o = np.argsort(-score); frac_t = np.cumsum(minutes[o]) / minutes.sum(); rec = np.cumsum(yte[o]) / yte.sum()
    return frac_t, rec
rng = np.random.default_rng(SEED)
bc = {'EN code-only risk': budget_curve(risk), 'RF code-only risk': budget_curve(risk_rf),
      'Random order': budget_curve(rng.random(len(te_t)))}
budget_tab = []
for nm, (ft, rc) in bc.items():
    row = {'ranking': nm}
    for B in [0.2, 0.3, 0.5]: row[f'recall_at_{int(B*100)}pct_farm_time'] = float(rc[np.searchsorted(ft, B)])
    budget_tab.append(row)
budget_tab = pd.DataFrame(budget_tab); budget_tab.round(4).to_csv(os.path.join(RES, 'table_profiling_budget.csv'), index=False)
premerge_prauc = {'EN': average_precision_score(yte, risk), 'RF': average_precision_score(yte, risk_rf)}
print(budget_tab.round(3))

# ------------------------------------------------------------------ 6. sensitivity: (C, l1 ratio) and CUSUM (k, h)
sens_rows = []
for C in [0.0003, 0.001, 0.003, 0.01, 0.03]:
    for l1 in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ap = []
        for tr, te in FOLDS:
            ap.append(average_precision_score(y[te], pred_en(fit_en(X_full.iloc[tr], y[tr], C, l1), X_full.iloc[te])))
        sens_rows.append(dict(lam=C, l1_ratio=l1, pr_auc_mean=np.mean(ap), pr_auc_std=np.std(ap, ddof=1)))
sens_en = pd.DataFrame(sens_rows); sens_en.round(4).to_csv(os.path.join(RES, 'table_sensitivity_en.csv'), index=False)
kh_rows = []
for k in [0.25, 0.5, 1.0, 1.5]:
    for h in [2.0, 4.0, 6.0]:
        cs_kh = add_cusum(d, k, h); dd = d.copy(); dd['S_lat'], dd['S_frame'] = cs_kh['S_lat'], cs_kh['S_frame']
        Xk = build_X(dd); f1s = []
        for tr, te in FOLDS[:3]:     # 3 folds to keep runtime short
            pin = inner_oof(Xk.iloc[tr], y[tr]); a, b, c = tune_gate(pin, dd['S_lat'].values[tr], y[tr])
            pk = pred_en(fit_en(Xk.iloc[tr], y[tr]), Xk.iloc[te]); f1s.append(f1_score(y[te], gate(pk, dd['S_lat'].values[te], a, b, c)))
        kh_rows.append(dict(k=k, h=h, f1_mean=np.mean(f1s)))
sens_kh = pd.DataFrame(kh_rows); sens_kh.round(4).to_csv(os.path.join(RES, 'table_sensitivity_cusum.csv'), index=False)
print(f'[sens] done {time.time()-T0:.1f}s')

# ------------------------------------------------------------------ 7. dataset summary + coefficients
ds = d.groupby('benchmark_scenario').agg(runs=('perf_regression', 'size'), reg_rate=('perf_regression', 'mean'),
                                         p50_median_ms=('p50_latency_ms', 'median'),
                                         rel_delta_reg=('r_lat', lambda s: float(np.median(np.exp(s[y[s.index] == 1]) - 1))),
                                         rel_delta_ok=('r_lat', lambda s: float(np.median(np.exp(s[y[s.index] == 0]) - 1))))
ds.round(4).to_csv(os.path.join(RES, 'table_dataset_summary.csv'))
mod_all = fit_en(X_full, y)
coef = pd.Series(mod_all[1].coef_[0], index=X_full.columns)
coef.sort_values(key=np.abs, ascending=False).round(4).to_csv(os.path.join(RES, 'table_en_coefficients.csv'), header=['coef_std'])

# ------------------------------------------------------------------ 8. figures
# Fig 2: PR curves (pooled out-of-fold scores)
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for k, lab, c, ls in [('prop', 'EN+CUSUM (proposed)', C_PROP, '-'), ('rf', 'Random forest', C_RF, '--'),
                      ('delta', 'Percent-delta score', C_DELTA, '-.'), ('fixed', 'Fixed-budget score', C_FIX, ':')]:
    pr, rc, _ = precision_recall_curve(y, oof[k])
    ax.plot(rc, pr, ls, color=c, lw=1.3, label=f'{lab} (AP {average_precision_score(y, oof[k]):.3f})')
for k, c, mk in [('prop', C_PROP, 'o'), ('rf', C_RF, 's'), ('delta', C_DELTA, '^'), ('fixed', C_FIX, 'D')]:
    a = oof_alarm[k]; ax.plot(recall_score(y, a), precision_score(y, a), mk, color=c, ms=5, mec='k', mew=0.5)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.02)
ax.legend(loc='lower left', frameon=True); savefig(fig, 'fig2_pr_curves')

# Fig 3: one stream over time - latency evidence, CUSUM statistic and alarms (2026-H1)
st = 'low_end_android|usage_chart_scroll'
ix = np.where((d['stream'] == st).values & (d['build_ts'] >= '2026-01-01').values)[0]
ixl = np.searchsorted(te_t, ix)
fig, axs = plt.subplots(2, 1, figsize=(3.4, 2.9), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
tt = d['build_ts'].values[ix]
axs[0].plot(tt, 100 * (np.exp(d['r_lat'].values[ix]) - 1), '-', color='0.55', lw=0.7)
axs[0].scatter(tt[y[ix] == 1], 100 * (np.exp(d['r_lat'].values[ix][y[ix] == 1]) - 1), s=22, facecolors='none',
               edgecolors='k', lw=0.8, label='true regression', zorder=3)
axs[0].scatter(tt[al_t[ixl]], 100 * (np.exp(d['r_lat'].values[ix][al_t[ixl]]) - 1), s=8, color=C_PROP, label='alarm (proposed)', zorder=4)
axs[0].set_ylim(-40, 110); axs[0].set_ylabel('p50 delta vs\nparent (%)'); axs[0].legend(loc='upper left', ncol=2, fontsize=7.8)
axs[1].plot(tt, d['S_lat'].values[ix], color=C_RF, lw=0.9, label='CUSUM $S_t$ (latency)')
axs[1].plot(tt, p_t[ixl] * 10, color=C_PROP, lw=0.7, alpha=0.8, label='EN prob. x10')
axs[1].axhline(g, color='k', ls=':', lw=0.8); axs[1].set_ylabel('statistic'); axs[1].legend(loc='upper left', ncol=2, fontsize=7.8)
fig.autofmt_xdate(rotation=30); axs[1].set_xlabel('Build date (2026)')
import matplotlib.dates as mdates; axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
savefig(fig, 'fig3_stream_timeline')

# Fig 4: elastic-net coefficients (top 14 by |coef|)
top = coef[coef != 0].sort_values(key=np.abs, ascending=False).head(14)[::-1]
NICE = {'z_r_lat': 'z latency ratio', 'z_m_frame': 'z frame p90', 'z_m_jank': 'z jank', 'z_m_mem': 'z memory',
        'z_m_cpu': 'z CPU/baseline', 'z_m_energy': 'z energy', 'z_m_bundle': 'z bundle', 'r_lat': 'log latency ratio',
        'S_lat': 'CUSUM S lat.', 'S_frame': 'CUSUM S frame', 'log_loc_add': 'log LOC added', 'static_main_thread_io_flag': 'main-thread I/O',
        'chart_library_upgraded': 'chart lib upgrade', 'new_deps': 'new dependencies', 'img_assets': 'image assets',
        'log_points': 'log meter points'}
SCN = {'bill_pdf_o': 'bill PDF', 'cold_start': 'cold start', 'dashboard_': 'dashboard', 'demand_res': 'DR opt-in',
       'outage_map': 'outage map', 'usage_char': 'chart scroll'}
PRE = {'chart_libr': 'chart lib', 'static_mai': 'main-thr. I/O', 'touches_ui': 'UI', 'touches_lo': 'DB'}
def nice(n):
    if n in NICE: return NICE[n]
    if n[:10] in PRE and n[10:11] == 'x': return PRE[n[:10]] + ' x ' + SCN.get(n[11:], n[11:])
    if n.startswith('sc_'): return 'scenario ' + n[3:].replace('_', ' ')
    return n.replace('chart_libr', 'chart').replace('static_mai', 'mainIO').replace('touches_ui', 'UI').replace('touches_lo', 'DB').replace('x', ' x ', 1).replace('_', ' ')
fig, ax = plt.subplots(figsize=(3.4, 2.7))
ax.barh([nice(n) for n in top.index], top.values, color=[C_PROP if v > 0 else C_RF for v in top.values])
ax.set_xlabel('Standardised coefficient (log-odds per s.d.)'); ax.tick_params(axis='y', labelsize=8.5)
savefig(fig, 'fig4_en_coefficients')

# Fig 5: sensitivity (left: PR-AUC over C x l1 ratio; right: F1 over CUSUM k x h)
fig, axs = plt.subplots(1, 2, figsize=(3.6, 2.0), gridspec_kw={'width_ratios': [1.3, 1]})
M = sens_en.pivot(index='lam', columns='l1_ratio', values='pr_auc_mean')
im = axs[0].imshow(M.values, cmap='Reds', aspect='auto')
axs[0].set_xticks(range(M.shape[1])); axs[0].set_xticklabels([f'{v:g}'.replace('0.', '.') for v in M.columns], fontsize=7.8)
axs[0].set_yticks(range(M.shape[0])); axs[0].set_yticklabels([f'{v:g}' for v in M.index], fontsize=7.8)
axs[0].set_xlabel('l1 ratio $\\rho$'); axs[0].set_ylabel('$\\lambda$'); axs[0].set_title('PR-AUC', fontsize=9)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.values[i, j]; dark = (v - M.values.min()) / (M.values.max() - M.values.min() + 1e-12) > 0.55
        axs[0].text(j, i, f'{1000 * v:.0f}', ha='center', va='center', fontsize=6.6, color='white' if dark else 'black')
M2 = sens_kh.pivot(index='k', columns='h', values='f1_mean')
axs[1].imshow(M2.values, cmap='Blues', aspect='auto')
axs[1].set_xticks(range(M2.shape[1])); axs[1].set_xticklabels([f'{v:g}' for v in M2.columns], fontsize=7.8)
axs[1].set_yticks(range(M2.shape[0])); axs[1].set_yticklabels([f'{v:g}' for v in M2.index], fontsize=7.8)
axs[1].set_xlabel('restart h'); axs[1].set_ylabel('slack k'); axs[1].set_title('F1 (gated)', fontsize=9)
for i in range(M2.shape[0]):
    for j in range(M2.shape[1]):
        v = M2.values[i, j]; dark = (v - M2.values.min()) / (M2.values.max() - M2.values.min() + 1e-12) > 0.55
        axs[1].text(j, i, f'{1000 * v:.0f}', ha='center', va='center', fontsize=6.6, color='white' if dark else 'black')
for a in axs: a.grid(False)
fig.tight_layout(w_pad=0.6); savefig(fig, 'fig5_sensitivity')

# Fig 6: profiling budget curve
fig, ax = plt.subplots(figsize=(3.3, 2.2))
for (nm, (ft, rc)), c, ls in zip(bc.items(), [C_PROP, C_RF, C_DELTA], ['-', '--', ':']):
    ax.plot(100 * ft, 100 * rc, ls, color=c, lw=1.3, label=nm)
ax.axvline(30, color='k', lw=0.6, ls=':')
ax.set_xlabel('Device-farm time spent (% of full profiling)'); ax.set_ylabel('Regressions profiled (%)')
ax.legend(loc='lower right'); savefig(fig, 'fig6_profiling_budget')

# Fig 1: flowchart (graphviz)
dot = os.path.join(HERE, 'flowchart.dot')
if shutil.which('dot') and os.path.exists(dot):
    subprocess.run(['dot', '-Tpng', '-Gdpi=300', dot, '-o', os.path.join(FIG, 'fig1_flowchart.png')], check=True)
    from PIL import Image
    Image.open(os.path.join(FIG, 'fig1_flowchart.png')).convert('RGB').save(os.path.join(RES, 'fig1_flowchart.jpg'), quality=92)

# ------------------------------------------------------------------ 9. metrics.json
out = dict(n_runs=int(len(d)), n_features=int(X_full.shape[1]), prevalence=float(y.mean()), missing_rate=miss,
           n_z_clipped=n_clipped, cusum=dict(k=K_CUSUM, h=H_CUSUM), en=dict(lam=EN_LAM, l1_ratio=EN_L1),
           cv_main={m: {c: float(main.loc[m, c]) for c in main.columns} for m in MORDER},
           cv_ablation={v: {c: float(abl.loc[v, c]) for c in abl.columns} for v in ABL},
           paired_tests=tests, mcnemar_prop_vs_rf=dict(prop_only_correct=b01, rf_only_correct=b10, p=mcn_p),
           temporal=temporal, temporal_sizes=dict(train=int(len(tr_t)), test=int(len(te_t)), test_pos=int(yte.sum())),
           temporal_gate=dict(tau_lo=tlo, tau_hi=thi, gamma=g),
           premerge_pr_auc=premerge_prauc, profiling_budget=budget_tab.set_index('ranking').to_dict('index'),
           tuned_mean=pd.DataFrame(tuned).mean(numeric_only=True).to_dict(),
           en_nonzero_full=int((coef != 0).sum()), runtime_s=round(time.time() - T0, 1))
json.dump(out, open(os.path.join(RES, 'metrics.json'), 'w'), indent=1, default=float)
print(f'[done] {time.time()-T0:.1f}s')
