"""Canonical project paths. Resolved once from the package root."""

from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[3]
CONFIG: Path = ROOT / "config"

DATA: Path = ROOT / "data"
DATA_RAW: Path = DATA / "raw"
DATA_INTERIM: Path = DATA / "interim"
DATA_PROCESSED: Path = DATA / "processed"
DATA_REFERENCE: Path = DATA / "reference"

OUTPUTS: Path = ROOT / "outputs"
OUT_FIGURES: Path = OUTPUTS / "figures"
OUT_TABLES: Path = OUTPUTS / "tables"
OUT_RESULTS: Path = OUTPUTS / "results"

PAPER: Path = ROOT / "paper"
DASHBOARD: Path = ROOT / "dashboard"


def ensure_dirs() -> None:
    for p in (
        DATA_RAW, DATA_INTERIM, DATA_PROCESSED, DATA_REFERENCE,
        OUT_FIGURES, OUT_TABLES, OUT_RESULTS,
    ):
        p.mkdir(parents=True, exist_ok=True)
