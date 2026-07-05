"""
visualization.py
-----------------
Reusable matplotlib plotting functions shared across notebooks, so every
notebook renders charts with the same visual language and none of them
duplicate plotting code.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {
    "pass": "#4FD1C5",
    "fail": "#F0654A",
    "neutral": "#7C8797",
    "accent": "#F2A93B",
    "grid": "#2A333F",
}

plt.rcParams.update({
    "figure.facecolor": "#0D1117",
    "axes.facecolor": "#0D1117",
    "axes.edgecolor": "#2A333F",
    "axes.labelcolor": "#E6EAF0",
    "text.color": "#E6EAF0",
    "xtick.color": "#7C8797",
    "ytick.color": "#7C8797",
    "grid.color": "#2A333F",
    "font.size": 10,
})


def plot_class_balance(pass_count: int, fail_count: int, ax=None):
    ax = ax or plt.gca()
    bars = ax.bar(["Pass", "Fail"], [pass_count, fail_count],
                   color=[PALETTE["pass"], PALETTE["fail"]])
    ax.set_title("Class Balance: Pass vs Fail Units")
    ax.set_ylabel("Unit count")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                 f"{int(b.get_height())}", ha="center", color="#E6EAF0")
    return ax


def plot_missingness(missing_report: pd.Series, threshold: float, top_n: int = 40, ax=None):
    ax = ax or plt.gca()
    top = missing_report.head(top_n)
    colors = [PALETTE["fail"] if v > threshold else PALETTE["neutral"] for v in top.values]
    ax.barh(range(len(top)), top.values, color=colors)
    ax.axvline(threshold, color=PALETTE["accent"], linestyle="--", linewidth=1,
                label=f"{threshold}% drop threshold")
    ax.set_yticks([])
    ax.set_xlabel("% missing")
    ax.set_title(f"Missing-value % — top {top_n} sensors")
    ax.legend(loc="lower right", frameon=False)
    return ax


def plot_feature_importance(importances: pd.Series, top_n: int = 15, ax=None):
    ax = ax or plt.gca()
    top = importances.head(top_n).sort_values()
    ax.barh([f"S{i}" for i in top.index], top.values, color=PALETTE["accent"])
    ax.set_xlabel("Random Forest importance")
    ax.set_title(f"Top {top_n} Failure-Driving Sensors")
    return ax


def plot_sensor_boxplot(pass_values, fail_values, sensor_name: str, ax=None):
    ax = ax or plt.gca()
    bp = ax.boxplot([pass_values, fail_values], labels=["Pass", "Fail"], patch_artist=True)
    for patch, color in zip(bp["boxes"], [PALETTE["pass"], PALETTE["fail"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(f"Sensor {sensor_name}: Pass vs Fail")
    return ax


def plot_roc_curve(fpr, tpr, roc_auc, ax=None):
    ax = ax or plt.gca()
    ax.plot(fpr, tpr, color=PALETTE["accent"], linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Fail Detection")
    ax.legend(loc="lower right", frameon=False)
    return ax


def plot_confusion_matrix(cm, ax=None):
    ax = ax or plt.gca()
    im = ax.imshow(cm, cmap="RdYlGn_r")
    labels = ["Pass", "Fail"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=13)
    ax.set_title("Confusion Matrix (Holdout Set)")
    return ax
