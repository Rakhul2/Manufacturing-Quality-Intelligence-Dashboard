"""
modeling.py
-----------
Model training and evaluation utilities for the SECOM pass/fail classification task.

Design decisions baked into this module (see docs/decision_log.md for full reasoning):
  - Random Forest is the primary model: handles nonlinear sensor interactions,
    is robust to unscaled/correlated features, and gives a built-in
    feature-importance ranking we can cross-check statistically.
  - Logistic Regression (with scaling) is kept as an interpretable baseline.
  - class_weight="balanced" is used instead of oversampling/SMOTE as the default,
    since the fail class is only ~6.6% of units and we want a first model that's
    cheap to explain; SMOTE is noted as an alternative in the decision log.
  - Stratified K-Fold is used everywhere because of the class imbalance.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support,
                              confusion_matrix, roc_curve, average_precision_score)


RANDOM_STATE = 42


def make_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.25):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def build_random_forest(n_estimators: int = 400, max_depth=None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_logistic_baseline() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
    )


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """
    Stratified K-fold cross-validation, scoring on ROC-AUC and average precision
    (both are appropriate for a highly imbalanced binary target; plain accuracy
    is not, since predicting "pass" for everything would already score ~93%).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "avg_precision": "average_precision"}
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "roc_auc_std": float(np.std(scores["test_roc_auc"])),
        "avg_precision_mean": float(np.mean(scores["test_avg_precision"])),
        "avg_precision_std": float(np.std(scores["test_avg_precision"])),
    }


def evaluate_on_holdout(model, X_train, y_train, X_test, y_test) -> dict:
    """Fit on train, evaluate on held-out test set. Returns metrics + arrays for plotting."""
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # y is coded -1 (pass) / 1 (fail); sklearn treats the larger label (1=fail) as positive
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)

    return {
        "model": model,
        "roc_auc": float(roc_auc),
        "avg_precision": float(avg_precision),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


def get_feature_importance(model, feature_names) -> pd.Series:
    """Return Random Forest feature importances as a sorted pandas Series."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False)
