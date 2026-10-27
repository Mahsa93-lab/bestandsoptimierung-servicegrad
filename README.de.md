# Bestandsoptimierung: Lieferfähigkeit vs. Kapitalbindung

[English version](README.md)

Ein Pharmagroßhändler muss fast jedes Medikament sofort liefern können – sonst bestellt die Apotheke beim
Wettbewerber. Gleichzeitig kostet jede Packung im Lager Kapital. Dieses kleine Modell zeigt mit **synthetischen
Daten**, wie man diesen Zielkonflikt mit Zahlen steuern kann:

1. **ABC-XYZ**: Welche Artikel bringen viel Wert, welche werden gleichmäßig oder unregelmäßig bestellt?
2. **Sicherheitsbestand und Meldebestand** pro Artikel und Niederlassung.
3. **Was kostet jedes Prozent Lieferfähigkeit?** Und: Bringt dasselbe Kapital mehr Lieferfähigkeit, wenn man es
   gezielt verteilt statt für alle Artikel denselben Servicegrad zu setzen?

> **Alle Daten sind synthetisch.** Sie sind so erzeugt, dass sie typische Muster zeigen (Saison, Wochentage,
> teure Spezialpräparate, mehrere Niederlassungen). Es sind keine Daten eines Unternehmens enthalten.

## Ergebnis in drei Sätzen

- **Mit demselben Kapital weniger Defekte:** Wird das Budget für Sicherheitsbestand gezielt verteilt statt
  „alle Artikel 97,5 %“, steigt die Lieferfähigkeit von 99,50 % auf 99,87 %, und der Anteil der Positionen mit
  Defekt sinkt von 0,75 % auf 0,24 % (**−69 %**).
- **Oder: gleiche Lieferfähigkeit mit weniger Kapital:** Schon mit dem Budget von „alle 95 %“ erreicht die gezielte
  Verteilung 99,69 % Lieferfähigkeit – mehr als die Referenz – bei **7,6 % weniger** Kapitalbindung.
- **Die letzten Prozente sind am teuersten:** Der Sicherheitsbestand steigt von 321 Tsd. € (95 %) auf 454 Tsd. €
  (99 %) und 602 Tsd. € (99,9 %). Die letzten 0,9 Prozentpunkte (99 → 99,9 %) kosten mehr als die vier davor (95 → 99 %).

![Kapital vs. Lieferfähigkeit](results/01_kapital_vs_lieferfaehigkeit.png)

![Die letzten Prozente sind am teuersten](results/02_letzte_prozente.png)

## Der Zielkonflikt, den das Modell sichtbar macht

![Wohin das Kapital wandert](results/04_kapital_nach_kategorie.png)

Mit einem einheitlichen Servicegrad stecken **58 % des Sicherheitskapitals in Spezialpräparaten**, obwohl sie nur
rund 0,3 % der Packungen ausmachen. Die Optimierung verschiebt Kapital zu den Schnelldrehern – dadurch sinkt die
Lieferfähigkeit der Spezialpräparate von 98,2 % auf 96,7 %.

Das ist **keine rein technische Frage, sondern eine Controlling-Entscheidung**: Das Modell zählt jede Packung gleich.
In der Praxis würde man für kritische oder teure Artikel einen höheren Mindest-Servicegrad festlegen, nach Rohertrag
gewichten oder diese Artikel zentral in einer Niederlassung lagern.

## ABC-XYZ und Saison

![ABC-XYZ](results/03_abc_xyz.png)

![Saison](results/05_saison.png)

## Wie das Modell funktioniert

| Schritt | Was passiert | Datei |
|---|---|---|
| Daten | 400 Artikel × 3 Niederlassungen = 1.200 SKUs, 2 Jahre, Montag bis Samstag. Nachfrage mit Saison (Erkältung im Winter, Allergie im Frühling), Wochentag (Montag hoch, Samstag niedrig) und Zufall (Gamma-Poisson). Lieferzeit des Herstellers 1–6 Arbeitstage. | `src/data.py` |
| ABC-XYZ | A/B/C nach Jahreswert (80 % / 95 %), X/Y/Z nach Variationskoeffizient der Wochennachfrage (≤ 0,5 / ≤ 1,0). | `src/classify.py` |
| Sicherheitsbestand | SB = z × σ<sub>Tag</sub> × √Lieferzeit; Meldebestand = Bedarf während der Lieferzeit + SB; Bestellmenge = Wochenbedarf. | `src/policy.py` |
| Verteilung | Start: alle Artikel auf 90 % (Mindestversorgung). Dann geht jeder weitere Euro dorthin, wo er die meisten Fehlmengen vermeidet (Greedy mit Fehlmengenfunktion), bis das Budget aufgebraucht ist. | `src/policy.py` |
| Simulation | Jahr 1 zum Lernen, Jahr 2 wird Tag für Tag simuliert: Bedarf und Schwankung rollierend (28 / 56 Tage), Bestellung bei Erreichen des Meldebestands, Defekt = Apotheke bestellt woanders (Menge verloren). | `src/simulate.py` |

**Kennzahlen:** *Lieferfähigkeit* = gelieferte / nachgefragte Packungen. *Positionen mit Defekt* = Anteil der
Artikel-Tage mit Nachfrage, an denen nicht alles geliefert werden konnte. *Kapitalbindung* = durchschnittlicher
Lagerbestand × Einkaufspreis.

## Grenzen (bewusst)

- **Normalverteilung:** Für Artikel mit sehr seltener Nachfrage (viele Nullen) ist sie ungenau; dort wären z. B.
  Croston oder eine empirische Verteilung besser.
- **Keine Feiertage, keine Lieferengpässe der Hersteller, keine Rabattvertragswechsel, kein Verfall.**
- **Zensierte Nachfrage:** In echten Daten sieht man bei einem Defekt die Nachfrage nicht in den Verkäufen –
  deshalb müsste man Bestellungen inklusive Defekte verwenden.
- **Gewichtung:** Jede Packung zählt gleich. Nach Rohertrag oder medizinischer Kritikalität zu gewichten, verändert
  die Verteilung (siehe Spezialpräparate).
- **Prognose:** Nur ein gleitender Mittelwert. Eine bessere Prognose senkt σ und damit direkt den Sicherheitsbestand –
  das wäre der nächste Hebel.

## Ausführen

```bash
pip install -r requirements.txt
python run_analysis.py   # ca. 1 Minute, schreibt alles nach results/
pytest -v
ruff check .
```

Ergebnisse: `results/results.md`, `results/summary.json`, `results/politiken.csv`, `results/kategorien.csv`,
`results/skus.csv`.

## Lizenz

MIT
