"""Synthetische Nachfragedaten für einen Pharma-Großhändler.

Alle Daten sind künstlich erzeugt (fester Seed). Sie bilden typische Muster nach:
wenige Schnelldreher, viele Langsamdreher, Saison (Erkältung, Allergie), Wochentage
(Montag hoch, Samstag niedrig) und sehr teure, selten bestellte Spezialpräparate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WORKDAYS_PER_WEEK = 6  # Montag bis Samstag
WEEKDAY_FACTOR = np.array([1.25, 1.05, 1.00, 1.00, 1.05, 0.55])  # Mo..Sa

# Monatsfaktoren (Jan..Dez)
FLAT = [1.0] * 12
COLD = [2.0, 1.9, 1.4, 1.0, 0.7, 0.5, 0.5, 0.5, 0.8, 1.2, 1.5, 1.9]
ALLERGY = [0.5, 0.6, 1.4, 2.2, 2.2, 2.0, 1.4, 0.9, 0.7, 0.5, 0.5, 0.5]
WINTER_MILD = [1.2, 1.2, 1.1, 1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 1.0, 1.1, 1.2]

# Kategorie: (Anzahl Artikel, EK-Median in EUR, EK-Streuung, Nachfrage-Median pro Tag und NL, Saison)
CATEGORIES = {
    "Schmerz & Fieber": (50, 4.0, 0.4, 6.0, WINTER_MILD),
    "Erkältung": (50, 6.0, 0.4, 4.0, COLD),
    "Allergie": (30, 8.0, 0.4, 2.5, ALLERGY),
    "Herz-Kreislauf": (80, 15.0, 0.5, 3.0, FLAT),
    "Diabetes": (50, 30.0, 0.5, 2.0, FLAT),
    "Antibiotika": (40, 12.0, 0.4, 1.5, WINTER_MILD),
    "Dermatologie": (60, 10.0, 0.5, 0.8, FLAT),
    "Spezialpräparate": (40, 1500.0, 0.6, 0.08, FLAT),
}
BRANCHES = {"NL Süd": 1.2, "NL Mitte": 1.0, "NL Nord": 0.8}
DISPERSION_SHAPE = 4.0  # Gamma-Poisson: kleinere Werte = unruhigere Nachfrage


@dataclass
class Dataset:
    skus: pd.DataFrame  # eine Zeile pro Artikel und Niederlassung
    demand: np.ndarray  # Tage x SKUs, Nachfrage in Packungen
    days: pd.DatetimeIndex  # Arbeitstage (Mo-Sa)


def workdays(start: str = "2024-10-28", weeks: int = 104) -> pd.DatetimeIndex:
    """Arbeitstage Montag bis Samstag (ohne Feiertage, zur Vereinfachung)."""
    all_days = pd.date_range(start, periods=weeks * 7, freq="D")
    return all_days[all_days.dayofweek < WORKDAYS_PER_WEEK]


def generate(seed: int = 42, weeks: int = 104) -> Dataset:
    rng = np.random.default_rng(seed)
    days = workdays(weeks=weeks)
    month_idx = days.month.to_numpy() - 1
    weekday_idx = days.dayofweek.to_numpy()

    rows, means = [], []
    article_id = 0
    for category, (n, ek_median, ek_sigma, dem_median, season) in CATEGORIES.items():
        season = np.asarray(season)
        for _ in range(n):
            article_id += 1
            ek = round(float(rng.lognormal(np.log(ek_median), ek_sigma)), 2)
            base = float(rng.lognormal(np.log(dem_median), 1.0))
            lead_time = int(rng.integers(1, 7))  # Wiederbeschaffungszeit in Arbeitstagen
            for branch, factor in BRANCHES.items():
                rows.append(
                    {
                        "artikel_id": f"A{article_id:04d}",
                        "kategorie": category,
                        "niederlassung": branch,
                        "ek_eur": ek,
                        "lieferzeit_tage": lead_time,
                    }
                )
                means.append(base * factor * season[month_idx] * WEEKDAY_FACTOR[weekday_idx])

    lam = np.column_stack(means)  # Tage x SKUs
    gamma = rng.gamma(DISPERSION_SHAPE, 1.0 / DISPERSION_SHAPE, size=lam.shape)
    demand = rng.poisson(lam * gamma).astype(np.int64)
    skus = pd.DataFrame(rows)
    skus.insert(0, "sku", skus["artikel_id"] + "-" + skus["niederlassung"].str.replace("NL ", "", regex=False))
    return Dataset(skus=skus, demand=demand, days=days)
