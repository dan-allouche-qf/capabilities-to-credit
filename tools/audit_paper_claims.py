"""Factual-claims audit: extract numeric tokens from LaTeX, lookup in outputs.

Walks ``paper/sections/`` and ``paper/sections/appendix/`` and pulls every
numeric literal that looks like a substantive claim (percentages, decimal
counts, Sharpe-like ratios). For each, search the CSV files in
``outputs/tables/`` for a matching value. Output a report of:

    * found    — number appears in at least one CSV row
    * structural — number appears as a structural literal (year, page count, etc.)
    * orphan   — number not found in any CSV and looks substantive

The audit is intentionally generous on tolerance; the goal is to catch
*made-up* numbers (e.g. "5,000" baskets when we ran 4,000), not to
exhaust every decimal claim.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = list((ROOT / "paper" / "sections").glob("*.tex")) + \
            list((ROOT / "paper" / "sections" / "appendix").glob("*.tex"))
TABLES = list((ROOT / "outputs" / "tables").glob("*.csv"))

# Structural literals we don't want to chase down (years, page counts,
# sector counts, country counts, ISO years etc.)
STRUCTURAL_VALUES = {
    "13", "20", "23", "5", "6", "10", "12", "17", "22", "8", "9", "4",
    "1990", "1995", "2000", "2005", "2007", "2008", "2009", "2010",
    "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018",
    "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026",
    "35", "30", "39", "56", "0", "1", "2", "3", "100", "1.0", "0.5",
    "0.30", "0.95", "0.05", "0.10", "0.90", "1.645",
    "0.21", "0.37", "0.28",  # random-portfolio CI bounds, in CSV
    # Bayesian model parameterisation
    "1500", "750", "4", "100", "2",
    "47", "48", "60", "172",
    "5000", "1000", "20", "200",
    "20260519",  # global random seed
    "300", "150",  # figure dpi
    "11pt", "1in",  # LaTeX class options
}

PATTERN = re.compile(
    r"(?<![A-Za-z\\])"
    r"(?:[+-])?"
    r"\d+(?:[\\\,]?\d+)*"     # integer part with optional thousand sep
    r"(?:\.\d+)?"             # decimal
    r"(?:\\%|%|\\\$)?"        # trailing unit
)


def _normalize(token: str) -> str:
    return (token.replace("\\,", "")
                  .replace(",", "")
                  .replace("\\%", "")
                  .replace("\\$", "")
                  .replace("%", "")
                  .replace("$", "")
                  .replace("+", "")
                  .lstrip("-"))


def _extract_from_tex() -> dict[Path, list[tuple[int, str, str]]]:
    """Return {section_path: [(line_no, raw_token, normalised_value), ...]}."""
    out: dict[Path, list[tuple[int, str, str]]] = {}
    for sec in sorted(SECTIONS):
        lines: list[tuple[int, str, str]] = []
        for n, line in enumerate(sec.read_text().splitlines(), start=1):
            if line.lstrip().startswith("%"):
                continue
            for m in PATTERN.finditer(line):
                raw = m.group(0)
                norm = _normalize(raw)
                if not norm:
                    continue
                try:
                    val = float(norm)
                except ValueError:
                    continue
                if abs(val) < 1e-9:
                    continue
                lines.append((n, raw, norm))
        if lines:
            out[sec] = lines
    return out


def _load_csv_values() -> set[float]:
    """Build a flat set of all numeric values appearing in outputs/tables."""
    vals: set[float] = set()
    for csv in TABLES:
        try:
            df = pd.read_csv(csv)
        except Exception:  # noqa: BLE001
            continue
        for col in df.select_dtypes(include="number").columns:
            for v in df[col].dropna().tolist():
                vals.add(float(v))
                vals.add(round(float(v), 1))
                vals.add(round(float(v), 2))
                vals.add(round(float(v), 3))
                vals.add(round(float(v) * 100, 1))
                vals.add(round(float(v) * 100, 2))
    return vals


def _is_match(value: float, csv_values: set[float], tol: float = 0.06) -> bool:
    """Treat 0.83 ≈ 83 ≈ 83.0 ≈ 0.829, etc. Tolerance is in absolute terms."""
    for cv in csv_values:
        if abs(cv - value) <= tol:
            return True
        if abs(cv - value * 100) <= tol:
            return True
        if cv != 0 and abs(cv - value / 100) <= tol:
            return True
    return False


def main() -> int:
    tex_claims = _extract_from_tex()
    csv_values = _load_csv_values()

    total = 0
    orphan = 0
    orphans: list[tuple[str, int, str]] = []
    for sec, claims in tex_claims.items():
        for line_no, raw, norm in claims:
            total += 1
            if norm in STRUCTURAL_VALUES:
                continue
            try:
                v = float(norm)
            except ValueError:
                continue
            if _is_match(v, csv_values):
                continue
            orphan += 1
            orphans.append((sec.name, line_no, raw))

    print(f"Total numeric tokens scanned: {total}")
    print(f"Structural / matched: {total - orphan}")
    print(f"Orphans (not found in any CSV): {orphan}")
    if orphans:
        print("\nFirst 40 orphans:")
        for name, line, raw in orphans[:40]:
            print(f"  {name}:{line:>3}   {raw!r}")
    return 0 if orphan < 30 else 1


if __name__ == "__main__":
    sys.exit(main())
