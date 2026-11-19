# A9P2 code - continuous performance profiling in mobile CI/CD

`run_all.py` runs the whole study end to end on one CPU core. It takes about 5 min on a lightly loaded machine and uses fixed seed 2026.

    pip install -r requirements.txt
    python3 run_all.py

It reads `../data/A9P2_energyapp_perf_profiling.csv` and writes `../results/` (CSV tables, `metrics.json`, JPG figures) and `../figures/` (PNG, 300 dpi, used in the paper).

## Pipeline

1. **Pre-processing**: runs are sorted by build time and split into 42 device-class x journey streams. Missing bundle size, frame time and energy are imputed with the median of the previous runs in the stream. The script computes the log latency ratio to the parent commit, then turns every metric into a rolling median/MAD robust z-score (W = 40 previous runs, clipped at +-8 to handle outliers). It also computes one-sided CUSUM statistics on the latency and frame-time z-scores (k = 0.5, restart at h = 4).
2. **Proposed method**: elastic-net logistic regression (lambda = 0.003, rho = 0.5) fitted by FISTA with adaptive restart (`ENLogit`), using 52 features (code, code x journey, z-scores, CUSUM). A two-key CUSUM gate decides the alarm: p >= tau_hi, or p >= tau_lo and S >= gamma. The gate is tuned on inner 3-fold out-of-fold scores of each training fold.
3. **Baselines**: fixed per-stream budgets on p50 latency and p90 frame time (quantile tuned on train), percent-delta rule against the parent commit (delta tuned on train), and a random forest on the same features (threshold tuned on out-of-bag scores).
4. **Evaluation**: stratified 5-fold CV (mean +- std), paired t-tests and McNemar's test, a temporal hold-out (train 2025, test 2026-H1), an ablation (8 variants), sensitivity grids for (lambda, rho) and CUSUM (k, h), and a pre-merge profiling-budget study using code-only risk.
5. **Fig. 1** is drawn from `flowchart.dot` with Graphviz.

## Outputs (`../results`)

| File | Content |
|---|---|
| `table_main_cv.csv` | Table III (CV): PR-AUC, ROC-AUC, F1, precision, recall, false-alarm rate, mean and std per method |
| `table_temporal.csv` | Table III (T columns): 2026-H1 temporal test |
| `table_ablation.csv` | Table IV |
| `table_dataset_summary.csv` | Table I |
| `table_sensitivity_en.csv`, `table_sensitivity_cusum.csv` | Fig. 5 grids |
| `table_profiling_budget.csv` | Fig. 6 recall at 20/30/50% device-farm time |
| `table_en_coefficients.csv` | Fig. 4 standardised coefficients |
| `cv_fold_metrics.csv`, `ablation_fold_metrics.csv`, `tuned_thresholds_per_fold.csv` | per-fold details |
| `metrics.json` | all headline numbers, statistical tests, runtime |
| `fig1..fig6 *.jpg` | figures (JPG copies) |
