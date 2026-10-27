"""ABC-XYZ-Analyse pro Artikel und Niederlassung."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import WORKDAYS_PER_WEEK

ABC_LIMITS = (0.80, 0.95)  # kumulierter Anteil am Jahreswert
XYZ_LIMITS = (0.5, 1.0)  # Variationskoeffizient der Wochennachfrage


def abc(values: np.ndarray) -> np.ndarray:
    """A = Artikel, die zusammen 80 % des Werts ausmachen, B = bis 95 %, C = Rest."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(-values, kind="stable")
    total = values.sum()
    cum_before = np.concatenate([[0.0], np.cumsum(values[order])[:-1]]) / total if total > 0 else np.zeros(len(values))
    classes = np.where(cum_before < ABC_LIMITS[0], "A", np.where(cum_before < ABC_LIMITS[1], "B", "C"))
    result = np.empty(len(values), dtype="<U1")
    result[order] = classes
    return result


def coefficient_of_variation(daily: np.ndarray) -> np.ndarray:
    """Variationskoeffizient (Std / Mittelwert) der Wochensummen; ohne Nachfrage = unendlich."""
    n_weeks = daily.shape[0] // WORKDAYS_PER_WEEK
    weekly = daily[: n_weeks * WORKDAYS_PER_WEEK].reshape(n_weeks, WORKDAYS_PER_WEEK, -1).sum(axis=1)
    mean = weekly.mean(axis=0)
    std = weekly.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mean > 0, std / mean, np.inf)
    return cv


def xyz(cv: np.ndarray) -> np.ndarray:
    """X = gleichmäßig, Y = schwankend, Z = unregelmäßig."""
    return np.where(cv <= XYZ_LIMITS[0], "X", np.where(cv <= XYZ_LIMITS[1], "Y", "Z"))


def classify(skus: pd.DataFrame, train_demand: np.ndarray) -> pd.DataFrame:
    out = skus.copy()
    out["jahresmenge"] = train_demand.sum(axis=0)
    out["jahreswert_eur"] = out["jahresmenge"] * out["ek_eur"]
    out["abc"] = abc(out["jahreswert_eur"].to_numpy())
    out["cv"] = coefficient_of_variation(train_demand)
    out["xyz"] = xyz(out["cv"].to_numpy())
    out["klasse"] = out["abc"] + out["xyz"]
    return out
