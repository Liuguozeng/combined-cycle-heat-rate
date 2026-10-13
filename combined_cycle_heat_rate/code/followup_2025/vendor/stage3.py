from __future__ import annotations
import numpy as np
import pandas as pd
ALPHA = 0.1

def winkler_score(actual: np.ndarray, prediction: np.ndarray, radius: float, alpha: float=ALPHA) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    lower = prediction - radius
    upper = prediction + radius
    score = np.full(len(actual), 2.0 * radius, dtype=float)
    score += np.where(actual < lower, 2.0 / alpha * (lower - actual), 0.0)
    score += np.where(actual > upper, 2.0 / alpha * (actual - upper), 0.0)
    return score

