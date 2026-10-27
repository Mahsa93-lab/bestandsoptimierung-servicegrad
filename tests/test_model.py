import numpy as np
import pytest

from src.classify import abc, coefficient_of_variation, xyz
from src.data import WORKDAYS_PER_WEEK, generate, workdays
from src.policy import (
    allocate_budget,
    expected_shortage_per_year,
    normal_loss,
    reorder_point,
    safety_stock,
    z_from_alpha,
)
from src.simulate import simulate


# ---------- Daten ----------
def test_workdays_are_monday_to_saturday():
    days = workdays(weeks=2)
    assert len(days) == 2 * WORKDAYS_PER_WEEK
    assert set(days.day_of_week) == set(range(6))


def test_generate_is_reproducible_and_non_negative():
    a, b = generate(seed=1, weeks=4), generate(seed=1, weeks=4)
    assert np.array_equal(a.demand, b.demand)
    assert (a.demand >= 0).all()
    assert a.demand.shape == (4 * WORKDAYS_PER_WEEK, len(a.skus))
    assert a.skus["sku"].is_unique


# ---------- ABC-XYZ ----------
def test_abc_limits():
    # sortiert: 50, 30, 10, 5, 3, 2 (Summe 100); Anteil davor: 0, 50, 80, 90, 95, 98 %
    # A solange davor < 80 %, B solange davor < 95 %
    values = np.array([10, 50, 2, 30, 5, 3])
    assert list(abc(values)) == ["B", "A", "C", "A", "B", "C"]


def test_abc_most_valuable_item_is_always_a():
    values = np.array([1.0, 1.0, 1000.0])
    assert abc(values)[2] == "A"


def test_xyz_thresholds():
    cv = np.array([0.1, 0.5, 0.51, 1.0, 1.5, np.inf])
    assert list(xyz(cv)) == ["X", "X", "Y", "Y", "Z", "Z"]


def test_coefficient_of_variation_constant_demand_is_zero():
    daily = np.full((WORKDAYS_PER_WEEK * 10, 2), 5)
    daily[:, 1] = 0
    cv = coefficient_of_variation(daily)
    assert cv[0] == pytest.approx(0.0)
    assert np.isinf(cv[1])


# ---------- Sicherheitsbestand ----------
@pytest.mark.parametrize("alpha, z", [(0.95, 1.645), (0.98, 2.054), (0.99, 2.326)])
def test_z_values(alpha, z):
    assert z_from_alpha(alpha) == pytest.approx(z, abs=1e-3)


def test_safety_stock_formula_and_monotonic():
    # z * sigma_Tag * sqrt(Lieferzeit) = 2 * 3 * 2 = 12
    assert float(safety_stock(2.0, 3.0, 4)) == pytest.approx(12.0)
    zs = np.linspace(0, 3, 10)
    ss = safety_stock(zs, np.full(10, 3.0), np.full(10, 4))
    assert np.all(np.diff(ss) > 0)


def test_reorder_point_is_lead_time_demand_plus_safety_stock():
    assert float(reorder_point(10.0, 3.0, 4, 2.0)) == pytest.approx(10 * 4 + 12)


def test_normal_loss_decreases_with_z():
    g = normal_loss(np.array([0.0, 1.0, 2.0, 3.0]))
    assert g[0] == pytest.approx(0.3989, abs=1e-4)
    assert np.all(np.diff(g) < 0)


def test_last_percent_is_most_expensive():
    """Jeder weitere Prozentpunkt Servicegrad kostet mehr Sicherheitsbestand als der vorige."""
    alphas = [0.95, 0.96, 0.97, 0.98, 0.99]
    ss = [float(safety_stock(z_from_alpha(a), 1.0, 1)) for a in alphas]
    assert np.all(np.diff(np.diff(ss)) > 0)


# ---------- Budgetverteilung ----------
@pytest.fixture
def small_portfolio():
    rng = np.random.default_rng(0)
    n = 40
    return dict(
        mean_daily=rng.uniform(1, 50, n),
        sd_daily=rng.uniform(0.5, 20, n),
        lead_time=rng.integers(1, 7, n),
        ek=rng.lognormal(2, 1.2, n),
    )


def test_allocate_budget_respects_budget_and_floor(small_portfolio):
    p = small_portfolio
    z_min = z_from_alpha(0.90)
    uniform = np.full(40, z_from_alpha(0.975))
    budget = float((safety_stock(uniform, p["sd_daily"], p["lead_time"]) * p["ek"]).sum())
    z = allocate_budget(budget, z_min=z_min, **p)
    spent = float((safety_stock(z, p["sd_daily"], p["lead_time"]) * p["ek"]).sum())
    assert spent <= budget + 1e-6
    assert np.all(z >= z_min - 1e-9)


def test_allocate_budget_beats_uniform_at_same_budget(small_portfolio):
    p = small_portfolio
    uniform = np.full(40, z_from_alpha(0.975))
    budget = float((safety_stock(uniform, p["sd_daily"], p["lead_time"]) * p["ek"]).sum())
    z = allocate_budget(budget, z_min=z_from_alpha(0.90), **p)
    args = (p["mean_daily"], p["sd_daily"], p["lead_time"])
    assert expected_shortage_per_year(z, *args).sum() < expected_shortage_per_year(uniform, *args).sum()


# ---------- Simulation ----------
def test_simulation_more_safety_stock_means_better_service_and_more_capital():
    ds = generate(seed=7, weeks=30)
    skus = ds.skus.iloc[:60]
    demand = ds.demand[:, :60]
    lead, ek = skus["lieferzeit_tage"].to_numpy(), skus["ek_eur"].to_numpy()
    start = 60
    low = simulate(demand, start, lead, np.full(60, z_from_alpha(0.80)), ek)
    high = simulate(demand, start, lead, np.full(60, z_from_alpha(0.99)), ek)
    assert 0 < low.fill_rate <= 1 and 0 < high.fill_rate <= 1
    assert high.fill_rate > low.fill_rate
    assert high.avg_capital_eur > low.avg_capital_eur
    assert np.all(low.served <= low.demanded)
