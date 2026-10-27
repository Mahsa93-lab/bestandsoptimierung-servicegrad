"""Sicherheitsbestand, Meldebestand und Verteilung des Bestandskapitals.

Annahmen (bewusst einfach): tägliche Nachfrage annähernd normalverteilt und unabhängig,
feste Wiederbeschaffungszeit, Bestellmenge = Bedarf einer Woche.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy.stats import norm

Z_GRID_STEP = 0.05
Z_MAX = 3.5


def z_from_alpha(alpha: float) -> float:
    """z-Wert für einen alpha-Servicegrad (Wahrscheinlichkeit, in einem Zyklus nicht auszuverkaufen)."""
    return float(norm.ppf(alpha))


def normal_loss(z: np.ndarray | float) -> np.ndarray:
    """Standard-Fehlmengenfunktion G(z): erwartete Fehlmenge pro Zyklus in Einheiten von sigma."""
    z = np.asarray(z, dtype=float)
    return norm.pdf(z) - z * (1.0 - norm.cdf(z))


def lead_time_sigma(sd_daily: np.ndarray, lead_time: np.ndarray) -> np.ndarray:
    return np.asarray(sd_daily, dtype=float) * np.sqrt(np.asarray(lead_time, dtype=float))


def safety_stock(z: np.ndarray | float, sd_daily: np.ndarray, lead_time: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(z, dtype=float), 0.0) * lead_time_sigma(sd_daily, lead_time)


def reorder_point(mean_daily, sd_daily, lead_time, z) -> np.ndarray:
    """Meldebestand = Bedarf während der Lieferzeit + Sicherheitsbestand."""
    return np.asarray(mean_daily) * np.asarray(lead_time) + safety_stock(z, sd_daily, lead_time)


def order_quantity(mean_daily: np.ndarray, days: int = 6) -> np.ndarray:
    return np.maximum(1.0, np.ceil(np.asarray(mean_daily, dtype=float) * days))


def expected_shortage_per_year(z, mean_daily, sd_daily, lead_time, days_per_year: int = 312) -> np.ndarray:
    """Erwartete Fehlmenge (Packungen) pro Jahr = Anzahl Bestellzyklen x Fehlmenge pro Zyklus."""
    q = order_quantity(mean_daily)
    cycles = np.asarray(mean_daily) * days_per_year / q
    return cycles * lead_time_sigma(sd_daily, lead_time) * normal_loss(z)


def allocate_budget(
    budget_eur: float,
    mean_daily: np.ndarray,
    sd_daily: np.ndarray,
    lead_time: np.ndarray,
    ek: np.ndarray,
    z_min: float,
) -> np.ndarray:
    """Greedy: Jeder weitere Euro Sicherheitsbestand geht dorthin, wo er die meisten Fehlmengen vermeidet.

    Start: alle Artikel auf z_min (Mindest-Servicegrad). Dann wird z in kleinen Schritten dort erhöht,
    wo das Verhältnis "vermiedene Fehlmenge pro zusätzlichem Euro" am größten ist – bis das Budget
    (Kapital im Sicherheitsbestand) aufgebraucht ist.
    """
    mean_daily, sd_daily = np.asarray(mean_daily, float), np.asarray(sd_daily, float)
    lead_time, ek = np.asarray(lead_time, float), np.asarray(ek, float)
    n = len(mean_daily)
    z = np.full(n, z_min)
    sigma_l = lead_time_sigma(sd_daily, lead_time)
    step_cost = Z_GRID_STEP * sigma_l * ek
    spent = float((safety_stock(z, sd_daily, lead_time) * ek).sum())

    def gain(i: int, zi: float) -> float:
        before = expected_shortage_per_year(zi, mean_daily[i], sd_daily[i], lead_time[i])
        after = expected_shortage_per_year(zi + Z_GRID_STEP, mean_daily[i], sd_daily[i], lead_time[i])
        return float(before - after)

    heap = []
    for i in range(n):
        if sigma_l[i] > 0 and step_cost[i] > 0:
            heapq.heappush(heap, (-gain(i, z[i]) / step_cost[i], i))
    while heap:
        neg_ratio, i = heapq.heappop(heap)
        if spent + step_cost[i] > budget_eur:
            continue
        z[i] += Z_GRID_STEP
        spent += step_cost[i]
        if z[i] + Z_GRID_STEP <= Z_MAX + 1e-9:
            heapq.heappush(heap, (-gain(i, z[i]) / step_cost[i], i))
    return z
