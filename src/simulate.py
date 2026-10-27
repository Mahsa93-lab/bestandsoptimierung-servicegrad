"""Tägliche Bestandssimulation für alle Artikel und Niederlassungen gleichzeitig.

Regel (s, S): Jeden Tag wird der Meldebestand s aus den letzten Wochen neu berechnet.
Fällt der disponible Bestand (Lager + offene Bestellungen) auf s oder darunter, wird bis
S = s + Wochenbedarf nachbestellt. Kann eine Apotheke nicht beliefert werden (Defekt),
bestellt sie woanders – die Menge ist verloren.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.policy import order_quantity, reorder_point

MEAN_WINDOW = 28  # Arbeitstage für den gleitenden Bedarf (Prognose)
SD_WINDOW = 56  # Arbeitstage für die Schwankung


@dataclass
class Result:
    fill_rate: float  # gelieferte / nachgefragte Packungen
    position_rate: float  # Anteil der Positionen (Artikel-Tag mit Nachfrage) ohne Defekt
    avg_capital_eur: float  # durchschnittliche Kapitalbindung im Lager
    served: np.ndarray  # pro SKU
    demanded: np.ndarray  # pro SKU
    capital_by_sku: np.ndarray  # Ø Kapital pro SKU


def _rolling(cum: np.ndarray, t: int, window: int) -> np.ndarray:
    return (cum[t + 1] - cum[t + 1 - window]) / window


def simulate(demand: np.ndarray, start: int, lead_time: np.ndarray, z: np.ndarray, ek: np.ndarray) -> Result:
    n_days, n = demand.shape
    lead_time = np.asarray(lead_time, dtype=int)
    cum = np.vstack([np.zeros((1, n)), np.cumsum(demand, axis=0)])
    cum2 = np.vstack([np.zeros((1, n)), np.cumsum(demand.astype(float) ** 2, axis=0)])

    def stats(t: int):
        mu = _rolling(cum, t, MEAN_WINDOW)
        m = _rolling(cum, t, SD_WINDOW)
        var = np.maximum(_rolling(cum2, t, SD_WINDOW) - m**2, 0.0) * SD_WINDOW / (SD_WINDOW - 1)
        return mu, np.sqrt(var)

    mu, sd = stats(start - 1)
    s = reorder_point(mu, sd, lead_time, z)
    on_hand = np.ceil(s + order_quantity(mu)).astype(float)
    on_order = np.zeros(n)
    arrivals = np.zeros((n_days + lead_time.max() + 1, n))
    cols = np.arange(n)

    served_sum = np.zeros(n)
    demand_sum = np.zeros(n)
    positions = defects = 0
    capital = np.zeros(n)
    for t in range(start, n_days):
        on_hand += arrivals[t]
        on_order -= arrivals[t]
        d = demand[t]
        served = np.minimum(on_hand, d)
        on_hand -= served
        served_sum += served
        demand_sum += d
        positions += int((d > 0).sum())
        defects += int((served < d).sum())

        mu, sd = stats(t)
        s = reorder_point(mu, sd, lead_time, z)
        target = s + order_quantity(mu)
        position = on_hand + on_order
        need = position <= s
        qty = np.where(need, np.ceil(target - position), 0.0)
        np.add.at(arrivals, (t + lead_time[need], cols[need]), qty[need])
        on_order += qty
        capital += on_hand * ek

    days = n_days - start
    return Result(
        fill_rate=float(served_sum.sum() / demand_sum.sum()),
        position_rate=float(1 - defects / positions) if positions else 1.0,
        avg_capital_eur=float(capital.sum() / days),
        served=served_sum,
        demanded=demand_sum,
        capital_by_sku=capital / days,
    )
