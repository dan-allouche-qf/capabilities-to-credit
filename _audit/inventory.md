# Inventory — capabilities-to-credit

## Identite & metadata
- Repo: `capabilities-to-credit` ("From Capabilities to Credit")
- Branche: `audit-2026-05` (tag `pre-audit-2026-05/main` = 7e9ada9 post phase 0)
- Auteur unique git: `Dan Allouche` (248489095+dan-allouche-qf@users.noreply.github.com)
- 220 fichiers trackes (le plus gros du portfolio)
- Python `>=3.11,<3.14`, package PEP 621 `newperformers` v0.1.0, license MIT (code) + CC-BY-4.0 (ratings CSV)

## Structure
- `src/newperformers/` (28 fichiers .py): `io/` (8 sources: WorldBank, WGI, WHO, SIPRI, IMF, FRED, yfinance, ratings), `etl/` (harmonize/impute/merge/schema/validate), `analysis/` (composite, composite_lasso, local_projections, lp_sensitivity, credit, credit_bayesian, granger, cointegration, synthetic_control, rating_change, survival, portfolio, portfolio_credit, robustness), `viz/` (figures_paper, palette, theme), `utils/` (config, logging, paths, seeds), `pipeline.py`
- `dashboard/` (7 fichiers): `streamlit_app.py` + 6 pages numerotees (country_profile, cross_country, credit_scorecard, portfolio, causal_evidence, rating_scorecard)
- `paper/main.pdf` (1.5 MB, 61 pages) — **mais paper/main.tex et paper/sections/*.tex NON TRACKES** (exclus par `.gitignore` regle `*.tex`)
- `outputs/figures/` = 47 PDF + 47 PNG ; `outputs/tables/` = 47 CSV
- `config/` (10 fichiers YAML/CSV/MD), `data/` (raw cache ignore), `tools/audit_paper_claims.py` + `tools/cross_check_ratings.py`
- `tests/` (10 fichiers, **39 tests** collectes)
- `Makefile`: targets `install / data / analysis / figures / paper / dashboard / tests / all / clean / clean-cache`

## Volumes disque (on-disk, .venv inclus)
- `outputs/` 15 MB | `data/` 2.1 MB | `paper/` 1.9 MB | `src/` 716 KB | `dashboard/` 72 KB | `.venv/` 952 MB (gitignored)

## Pre-audit security scan
- trufflehog 22 raw hits, **verified=0**, tous dans `.venv/` (vendored pandas/pip/urllib3/pyarrow). Aucun secret commite.
- abs_paths.txt vide pour le source (1 hit dans `paper/main.log` artefact LaTeX non versionne).

## Cas identite (Phase 7)
- **Cas B (no-op)** : "Dan Allouche" present + normalise dans `pyproject.toml`, `README.md`, `src/newperformers/__init__.py`, `dashboard/streamlit_app.py`, `paper/main.tex (on-disk uniquement)`. Aucune variante "Dan Joseph Allouche" / "D. Allouche". Aucune mention IA.

## Notes
- README claims "47 figures, 39 pytest tests" verifies exactement.
- `tools/audit_paper_claims.py` (auditeur factuel maison) : 1139 tokens numeriques scannes, **0 orphan**.
- `tools/cross_check_ratings.py` : 17/17 transitions de rating verifiees.
