PYTHON ?= python
# The src/ layout is added to PYTHONPATH explicitly because the editable .pth
# file occasionally fails to load on iCloud-backed paths with spaces.
export PYTHONPATH := $(CURDIR)/src
PIPELINE := $(PYTHON) -m newperformers.pipeline

.PHONY: help install data analysis figures paper dashboard tests all clean clean-cache

help:
	@echo "Targets:"
	@echo "  install     - editable install with dev extras"
	@echo "  data        - fetch + harmonize + impute panel data"
	@echo "  analysis    - composite, local projections, credit, portfolio"
	@echo "  figures     - render the 28 paper PDF figures + plotly HTML"
	@echo "  paper       - compile paper/main.tex via latexmk"
	@echo "  dashboard   - launch the Streamlit dashboard locally"
	@echo "  tests       - run the pytest suite"
	@echo "  all         - data -> analysis -> figures -> paper"
	@echo "  clean       - remove build artifacts (keeps the data cache)"
	@echo "  clean-cache - also remove data/{raw,interim,processed}"

install:
	$(PYTHON) -m pip install -e ".[dev,synth]"

data:
	$(PIPELINE) data

analysis: data
	$(PIPELINE) analysis

figures: analysis
	$(PIPELINE) figures

paper: figures
	cd paper && pdflatex -interaction=nonstopmode main.tex
	cd paper && $(HOME)/Library/TinyTeX/bin/universal-darwin/biber main || biber main
	cd paper && pdflatex -interaction=nonstopmode main.tex
	cd paper && pdflatex -interaction=nonstopmode main.tex

dashboard:
	cd dashboard && streamlit run streamlit_app.py

tests:
	$(PYTHON) -m pytest

all: paper

clean:
	rm -rf outputs/figures/*.pdf outputs/figures/*.png outputs/figures/*.html
	rm -rf outputs/tables/*.csv outputs/tables/*.tex
	cd paper && latexmk -C

clean-cache: clean
	rm -rf data/raw/* data/interim/* data/processed/*
