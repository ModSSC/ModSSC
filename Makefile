.DEFAULT_GOAL := help

VENV_PY := .venv/bin/python
PYTHON ?= $(if $(wildcard $(VENV_PY)),$(VENV_PY),python)

.PHONY: help install-dev install-docs lint format test docs-serve docs-build build check-dist release-prep clean

help:
	@echo "Targets:"
	@echo "  install-dev   Install dev dependencies"
	@echo "  install-docs  Install docs dependencies"
	@echo "  format        Format with Ruff"
	@echo "  lint          Lint with Ruff"
	@echo "  test          Run pytest"
	@echo "  docs-serve    Serve docs locally (MkDocs)"
	@echo "  docs-build    Build docs (MkDocs)"
	@echo "  build         Build sdist/wheel with build"
	@echo "  check-dist    Validate dist metadata with twine"
	@echo "  release-prep  Build + twine check (no publish)"

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

install-docs:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[docs]"

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest

docs-serve:
	$(PYTHON) -m mkdocs serve

docs-build:
	$(PYTHON) -m mkdocs build

build:
	$(PYTHON) -m build

check-dist:
	$(PYTHON) -m twine check dist/*

release-prep: build check-dist

clean:
	rm -rf dist build htmlcov .coverage .pytest_cache .ruff_cache cache artifacts data outputs runs site
