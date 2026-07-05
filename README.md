# Manufacturing Quality Intelligence Dashboard – End-to-End Failure Diagnostics

**Status:** analysis foundation (notebooks 01–07 + `src/`) complete and executed. Dashboard rebuilt from real pipeline artifacts (`dashboard/index.html`). `docs/` (methodology, decision log, business recommendations) and the recruiter-facing top-level README story are next.

## What's here

- `notebooks/` — the full 01–07 analytical narrative: business understanding → data understanding → cleaning → EDA → feature engineering → model building → statistical validation. Every notebook is executed with real output (numbers, charts).
- `src/` — reusable pipeline code (`preprocessing.py`, `statistics.py`, `modeling.py`, `visualization.py`), imported by the notebooks rather than duplicated.
- `data/processed/` — cleaned dataset and every derived artifact (feature importances, model metrics, statistical validation results, the assembled `dashboard_data.json`) that the dashboard is built from — no hardcoded numbers.
- `dashboard/index.html` — single-file interactive dashboard (Plotly) with a **Dashboard View / Interview Mode** toggle. Interview Mode surfaces the reasoning behind each design/analysis decision inline.

## Headline finding

Of the top 15 sensors ranked by Random Forest importance, **7 are independently confirmed** by a Mann-Whitney U test with Benjamini-Hochberg FDR correction across all 442 cleaned sensors — that validated shortlist (not the raw top-15) is what's handed to engineering.

## Model honesty

Random Forest holdout ROC-AUC: **0.816**. At the default 0.5 threshold, recall on the fail class is 0 (expected given a ~6.6% fail rate) — a tuned threshold (0.105, chosen to maximize F1) gives 57.7% recall / 27.8% precision on failures. This trade-off is reported explicitly rather than hidden behind a single accuracy number.
