# A9P2 - Continuous Performance Profiling in Mobile CI/CD Pipelines for Early Detection of Application Performance Regressions

Author: Mug. ICE2CPT 2026, Track 1. Application: smart-grid consumer energy-monitoring app.
Proposed method: elastic-net logistic regression with CUSUM change-point gating, on stream-normalised device-farm profiling evidence.
Baselines: fixed thresholds, percent-delta rule, random forest.

Contents:
- `paper/`: IEEE paper (.docx and .pdf, 6 pages)
- `data/`: synthetic dataset (12,000 CI profiling runs) and its README / data dictionary
- `code/`: `run_all.py` (whole pipeline), `flowchart.dot`, `requirements.txt`, `README.md`
- `results/`: all figures (JPG), all tables (CSV), `metrics.json`

Reproduce:

    cd code && pip install -r requirements.txt
    python3 run_all.py   # writes ../results and ../figures (~5 min, 1 core)
