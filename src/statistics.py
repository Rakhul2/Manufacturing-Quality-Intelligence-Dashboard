"""
statistics.py
-------------
Statistical validation utilities. We use these to check whether sensors that
a model flags as "important" actually show a statistically significant
distributional difference between pass and fail units, and to correct for
the fact that we are testing hundreds of sensors at once (multiple comparisons).
"""

import numpy as np
import pandas as pd
from scipy import stats


def mann_whitney_test(pass_values: np.ndarray, fail_values: np.ndarray) -> dict:
    """
    Run a Mann-Whitney U test comparing the pass and fail distributions for one sensor.

    We use Mann-Whitney U (not a t-test) because sensor readings are not guaranteed
    to be normally distributed and the fail group is small (~100 units), so a
    non-parametric rank-based test is more robust.
    """
    pass_values = pass_values[~np.isnan(pass_values)]
    fail_values = fail_values[~np.isnan(fail_values)]
    if len(pass_values) < 3 or len(fail_values) < 3:
        return {"u_stat": np.nan, "p_value": np.nan, "n_pass": len(pass_values), "n_fail": len(fail_values)}
    u_stat, p_value = stats.mannwhitneyu(pass_values, fail_values, alternative="two-sided")
    return {"u_stat": float(u_stat), "p_value": float(p_value),
            "n_pass": len(pass_values), "n_fail": len(fail_values)}


def benjamini_hochberg(p_values: pd.Series, fdr: float = 0.05) -> pd.Series:
    """
    Benjamini-Hochberg FDR correction for multiple comparisons.
    Returns a boolean Series indexed the same as p_values: True = significant after correction.

    Why this matters: testing 442 sensors independently at alpha=0.05 would give
    ~22 false positives by chance alone. BH control keeps the false discovery
    rate at the target level while still allowing detection of real effects.
    """
    p = p_values.dropna().sort_values()
    n = len(p)
    ranks = np.arange(1, n + 1)
    thresholds = ranks / n * fdr
    below = p.values <= thresholds
    if not below.any():
        sig_index = []
    else:
        max_rank = np.max(np.where(below)[0])
        sig_index = p.index[: max_rank + 1]
    result = pd.Series(False, index=p_values.index)
    result.loc[sig_index] = True
    return result


def evaluate_sensors(df: pd.DataFrame, sensor_cols: list, label_col: str = "Pass/Fail",
                      fdr: float = 0.05) -> pd.DataFrame:
    """
    Run Mann-Whitney U tests for every sensor column against the pass/fail label,
    apply Benjamini-Hochberg FDR correction, and return a results DataFrame sorted
    by p-value with a 'significant' boolean column.
    """
    fail_mask = df[label_col] == 1
    pass_mask = df[label_col] == -1

    rows = []
    for col in sensor_cols:
        res = mann_whitney_test(df.loc[pass_mask, col].values, df.loc[fail_mask, col].values)
        res["sensor"] = col
        rows.append(res)

    results = pd.DataFrame(rows).set_index("sensor")
    results["significant"] = benjamini_hochberg(results["p_value"], fdr=fdr)
    return results.sort_values("p_value")
