from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


@dataclass
class MLPConfig:
    hidden_layer_sizes: Tuple[int, ...] = (64, 32)
    activation: str = "relu"
    alpha: float = 1e-4
    learning_rate_init: float = 1e-3
    max_iter: int = 200
    early_stopping: bool = True
    random_state: int = 42


def build_mlp(cfg: Optional[MLPConfig] = None):
    """Return a scikit-learn pipeline: StandardScaler -> MLPClassifier."""
    if cfg is None:
        cfg = MLPConfig()

    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    mlp = MLPClassifier(
        hidden_layer_sizes=cfg.hidden_layer_sizes,
        activation=cfg.activation,
        alpha=cfg.alpha,
        learning_rate_init=cfg.learning_rate_init,
        max_iter=cfg.max_iter,
        early_stopping=cfg.early_stopping,
        random_state=cfg.random_state,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", mlp),
    ])
    return pipe


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute basic metrics for a trained model.

    Returns a dict with: accuracy, f1_weighted, roc_auc (binary), and per-class report.
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

    y_pred = model.predict(X_test)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        pass

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "report": classification_report(y_test, y_pred, digits=3),
    }
    # Binary ROC-AUC if possible
    if y_proba is not None and len(np.unique(y_test)) == 2:
        try:
            # assume proba for class 1
            pos_idx = list(model.classes_).index(1)
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, pos_idx]))
        except Exception:
            pass
    return metrics
