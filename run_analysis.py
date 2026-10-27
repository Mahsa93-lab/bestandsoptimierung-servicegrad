"""Gesamte Analyse: Daten erzeugen, ABC-XYZ, Politiken simulieren, Grafiken und Ergebnisse schreiben.

Aufruf: python run_analysis.py   (Laufzeit ca. 1 Minute)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.classify import classify
from src.data import WORKDAYS_PER_WEEK, generate
from src.policy import allocate_budget, safety_stock, z_from_alpha
from src.simulate import simulate

OUT = Path("results")
ALPHAS = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]
ALPHA_MIN = 0.90  # Mindest-Servicegrad für jeden Artikel (Versorgung)
REFERENCE_ALPHA = 0.975  # heutige Regel im Beispiel: alle Artikel 97,5 %

# Farben (validierte Referenzpalette, helle Fläche)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
UNIFORM, OPTIMIZED = "#eb6834", "#2a78d6"
BLUE_RAMP = ["#eef4fc", "#cfe0f6", "#9fc2ee", "#5f9be2", "#2a78d6", "#1d5aa5", "#123c70"]


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f} %".replace(".", ",")


def eur(x: float) -> str:
    return f"{x:,.0f} €".replace(",", ".")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ds = generate()
    demand = ds.demand
    train = demand.shape[0] // 2  # Jahr 1: Parameter lernen, Jahr 2: simulieren
    skus = classify(ds.skus, demand[:train])
    mu = demand[:train].mean(axis=0)
    sd = demand[:train].std(axis=0, ddof=1)
    lead, ek = skus["lieferzeit_tage"].to_numpy(), skus["ek_eur"].to_numpy()
    z_min = z_from_alpha(ALPHA_MIN)

    rows = []
    z_by_policy = {}
    for alpha in ALPHAS:
        z_uni = np.full(len(mu), z_from_alpha(alpha))
        budget = float((safety_stock(z_uni, sd, lead) * ek).sum())
        z_opt = allocate_budget(budget, mu, sd, lead, ek, z_min)
        for name, z in (("einheitlich", z_uni), ("optimiert", z_opt)):
            r = simulate(demand, train, lead, z, ek)
            rows.append(
                {
                    "regel": name,
                    "alpha_referenz": alpha,
                    "budget_sicherheitsbestand_eur": budget,
                    "lieferfaehigkeit": r.fill_rate,
                    "positionen_ohne_defekt": r.position_rate,
                    "kapitalbindung_eur": r.avg_capital_eur,
                }
            )
            z_by_policy[(name, alpha)] = (z, r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "politiken.csv", index=False)

    uni = res[res.regel == "einheitlich"].set_index("alpha_referenz")
    opt = res[res.regel == "optimiert"].set_index("alpha_referenz")

    # Vergleich 1: gleiches Budget wie "alle 97,5 %"
    u_ref, o_ref = uni.loc[REFERENCE_ALPHA], opt.loc[REFERENCE_ALPHA]
    defect_u, defect_o = 1 - u_ref.positionen_ohne_defekt, 1 - o_ref.positionen_ohne_defekt
    # Vergleich 2: kleinste optimierte Variante, die mindestens die Lieferfähigkeit der Referenz erreicht
    reach = opt[opt.lieferfaehigkeit >= u_ref.lieferfaehigkeit].sort_values("kapitalbindung_eur").iloc[0]
    saving = 1 - reach.kapitalbindung_eur / u_ref.kapitalbindung_eur

    # Wo das Kapital hingeht (Referenzbudget)
    z_u, r_u = z_by_policy[("einheitlich", REFERENCE_ALPHA)]
    z_o, r_o = z_by_policy[("optimiert", REFERENCE_ALPHA)]
    detail = skus.assign(
        sb_einheitlich=safety_stock(z_u, sd, lead) * ek,
        sb_optimiert=safety_stock(z_o, sd, lead) * ek,
        geliefert_einheitlich=r_u.served,
        geliefert_optimiert=r_o.served,
        nachfrage=r_u.demanded,
    )
    by_cat = detail.groupby("kategorie")[
        ["sb_einheitlich", "sb_optimiert", "geliefert_einheitlich", "geliefert_optimiert", "nachfrage"]
    ].sum()
    by_cat["lf_einheitlich"] = by_cat.geliefert_einheitlich / by_cat.nachfrage
    by_cat["lf_optimiert"] = by_cat.geliefert_optimiert / by_cat.nachfrage
    by_cat = by_cat.sort_values("sb_einheitlich", ascending=False)
    by_cat.to_csv(OUT / "kategorien.csv")
    detail.to_csv(OUT / "skus.csv", index=False)

    # Grafik 1: Kapital gegen Lieferfähigkeit
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    style(
        ax,
        "Mehr Lieferfähigkeit für dasselbe Kapital",
        "Ø Kapitalbindung im Lager (Tsd. €)",
        "Lieferfähigkeit (Packungen, %)",
    )
    ax.plot(
        uni.kapitalbindung_eur / 1e3,
        uni.lieferfaehigkeit * 100,
        "-o",
        color=UNIFORM,
        lw=2,
        ms=8,
        label="Einheitlicher Servicegrad für alle Artikel",
    )
    ax.plot(
        opt.kapitalbindung_eur / 1e3,
        opt.lieferfaehigkeit * 100,
        "-o",
        color=OPTIMIZED,
        lw=2,
        ms=8,
        label="Kapital dorthin, wo es am meisten Defekte vermeidet",
    )
    for a, row in uni.iterrows():
        ax.annotate(
            f"alle {a * 100:g} %".replace(".", ","),
            (row.kapitalbindung_eur / 1e3, row.lieferfaehigkeit * 100),
            textcoords="offset points",
            xytext=(6, -14),
            fontsize=8,
            color=INK2,
        )
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "01_kapital_vs_lieferfaehigkeit.png", dpi=160)
    plt.close(fig)

    # Grafik 2: Die letzten Prozente sind am teuersten (analytisch, Sicherheitsbestand)
    grid = np.array([0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999])
    sb = np.array([(safety_stock(np.full(len(mu), z_from_alpha(a)), sd, lead) * ek).sum() for a in grid])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    style(
        ax,
        "Die letzten Prozente sind am teuersten",
        "Servicegrad (alpha) für alle Artikel",
        "Kapital im Sicherheitsbestand (Tsd. €)",
    )
    ax.plot(range(len(grid)), sb / 1e3, "-o", color=OPTIMIZED, lw=2, ms=8)
    ax.set_xticks(range(len(grid)), [f"{a * 100:g} %".replace(".", ",") for a in grid])
    for i in (2, 3, 5, 7):
        ax.annotate(
            f"{sb[i] / 1e3:,.0f}".replace(",", "."),
            (i, sb[i] / 1e3),
            textcoords="offset points",
            xytext=(-8, 8),
            fontsize=9,
            color=INK,
        )
    fig.tight_layout()
    fig.savefig(OUT / "02_letzte_prozente.png", dpi=160)
    plt.close(fig)

    # Grafik 3: ABC-XYZ-Matrix (Anzahl SKUs und Anteil am Jahreswert)
    total = skus.jahreswert_eur.sum()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.figure.set_facecolor(SURFACE)
    share = np.zeros((3, 3))
    count = np.zeros((3, 3), dtype=int)
    for i, a in enumerate("ABC"):
        for j, x in enumerate("XYZ"):
            m = (skus.abc == a) & (skus.xyz == x)
            share[i, j] = skus.loc[m, "jahreswert_eur"].sum() / total
            count[i, j] = int(m.sum())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=share.max())
    for i in range(3):
        for j in range(3):
            dark = share[i, j] > share.max() * 0.55
            ax.text(
                j,
                i,
                f"{count[i, j]} SKUs\n{pct(share[i, j], 1)} Wert",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if dark else INK,
            )
    ax.set_xticks(range(3), ["X (gleichmäßig)", "Y (schwankend)", "Z (unregelmäßig)"], color=INK2)
    ax.set_yticks(range(3), ["A (80 % Wert)", "B (15 %)", "C (5 %)"], color=INK2)
    ax.set_title("ABC-XYZ: Artikel x Niederlassung", loc="left", fontsize=12, fontweight="bold", color=INK, pad=12)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "03_abc_xyz.png", dpi=160)
    plt.close(fig)

    # Grafik 4: Wohin das Kapital wandert (gleiches Budget)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    style(ax, "Wohin das Sicherheitskapital wandert (gleiches Budget)", "", "Kapital im Sicherheitsbestand (Tsd. €)")
    x = np.arange(len(by_cat))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, by_cat.sb_einheitlich / 1e3, w, color=UNIFORM, label="Einheitlich 97,5 %")
    ax.bar(x + w / 2 + 0.01, by_cat.sb_optimiert / 1e3, w, color=OPTIMIZED, label="Optimiert")
    ax.set_xticks(x, by_cat.index, rotation=25, ha="right", color=INK2)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "04_kapital_nach_kategorie.png", dpi=160)
    plt.close(fig)

    # Grafik 5: Saison
    weeks = demand.shape[0] // WORKDAYS_PER_WEEK
    weekly = demand[: weeks * WORKDAYS_PER_WEEK].reshape(weeks, WORKDAYS_PER_WEEK, -1).sum(axis=1)
    week_start = ds.days[::WORKDAYS_PER_WEEK][:weeks]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    style(ax, "Saison: Nachfrage pro Woche (alle Niederlassungen)", "", "Packungen pro Woche (Tsd.)")
    for cat, color in (("Erkältung", OPTIMIZED), ("Allergie", UNIFORM), ("Herz-Kreislauf", "#1baf7a")):
        m = (skus.kategorie == cat).to_numpy()
        ax.plot(week_start, weekly[:, m].sum(axis=1) / 1e3, color=color, lw=2, label=cat)
    ax.set_ylim(0, 16.5)
    ax.legend(frameon=False, labelcolor=INK, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT / "05_saison.png", dpi=160)
    plt.close(fig)

    summary = {
        "skus": int(len(skus)),
        "artikel": int(skus.artikel_id.nunique()),
        "niederlassungen": int(skus.niederlassung.nunique()),
        "simulierte_tage": int(demand.shape[0] - train),
        "referenz": {
            "regel": f"alle Artikel alpha = {REFERENCE_ALPHA}",
            "lieferfaehigkeit": u_ref.lieferfaehigkeit,
            "defektquote_positionen": defect_u,
            "kapitalbindung_eur": u_ref.kapitalbindung_eur,
        },
        "optimiert_gleiches_budget": {
            "lieferfaehigkeit": o_ref.lieferfaehigkeit,
            "defektquote_positionen": defect_o,
            "kapitalbindung_eur": o_ref.kapitalbindung_eur,
            "defekte_weniger": 1 - defect_o / defect_u,
        },
        "optimiert_gleiche_lieferfaehigkeit": {
            "budget_wie_alpha": float(reach.name),
            "lieferfaehigkeit": reach.lieferfaehigkeit,
            "kapitalbindung_eur": reach.kapitalbindung_eur,
            "kapital_weniger": saving,
        },
        "sicherheitsbestand_kapital": {f"{a}": float(v) for a, v in zip(grid, sb)},
        "anteil_spezialpraeparate_am_sicherheitskapital_einheitlich": float(
            by_cat.loc["Spezialpräparate", "sb_einheitlich"] / by_cat.sb_einheitlich.sum()
        ),
        "lieferfaehigkeit_spezialpraeparate": {
            "einheitlich": float(by_cat.loc["Spezialpräparate", "lf_einheitlich"]),
            "optimiert": float(by_cat.loc["Spezialpräparate", "lf_optimiert"]),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Ergebnisse (automatisch erzeugt)",
        "",
        f"{summary['skus']} SKUs ({summary['artikel']} Artikel x {summary['niederlassungen']} Niederlassungen), "
        f"{summary['simulierte_tage']} simulierte Arbeitstage. **Alle Daten sind synthetisch.**",
        "",
        "| Regel | Lieferfähigkeit | Positionen mit Defekt | Ø Kapitalbindung |",
        "|---|---|---|---|",
    ]
    for _, r in res.iterrows():
        label = f"{r.regel}, Budget wie alle {r.alpha_referenz * 100:g} %".replace(".", ",")
        defekt = pct(1 - r.positionen_ohne_defekt)
        lines.append(f"| {label} | {pct(r.lieferfaehigkeit)} | {defekt} | {eur(r.kapitalbindung_eur)} |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
