# Load .env file if it exists, create from example if not
ifeq (,$(wildcard .env))
    $(shell cp .env.example .env 2>/dev/null || true)
endif
ifneq (,$(wildcard .env))
    include .env
    export
endif

PYTHON_CMD := $(PYTHON_INTERPRETER)$(PYTHON_VERSION)

## ---------------
## Global commands
## ---------------

.DEFAULT_GOAL := help

help:	## Self Documenting Commands
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

install:	## Set up Python interpreter environment and install dependencies
	$(PYTHON_CMD) -m venv .venv
	@echo ">>> Virtual environment created."
	@echo ">>> Installing dependencies..."
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e .
	@echo "Installation complete. Virtual env activated"
	@echo "Activate virtual env:"
	@echo "source .venv/bin/activate"
	
doc: # Generate the static documentation (run `mkdocs serve` after).
	mkdocs build

doclive: # Generate the static documentation (run `mkdocs serve` after).
	mkdocs serve

clean: ## Delete all compiled Python files
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete	

lint: ## Lint using ruff (use `make format` to do formatting)
	ruff format --check
	ruff check

format: ## Format source code with ruff
	ruff check --fix
	ruff format

test: ## Run tests
	@pytest tests

.PHONY: data
data: ## Dataset operations
	@python -m datasets
