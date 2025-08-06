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

## ------------------
## Data pipeline commands
## ------------------

extract: ## Extract data for specific entity (usage: make extract ENTITY=project)
	python main.py extract $(ENTITY)

extract-all: ## Extract all datasets
	python main.py extract-all

transform: ## Transform specific entity (usage: make transform ENTITY=project)  
	python main.py transform $(ENTITY)

transform-all: ## Transform all entities
	python main.py transform-all

load: ## Load processed data to cloud (usage: make load ENTITY=project)
	python main.py load $(ENTITY)

load-all: ## Load all processed files to BigQuery
	python main.py load-all-processed-to-bq

pipeline: ## Full ETL for a specific entity (usage: make pipeline ENTITY=project)  
	python main.py pipeline $(ENTITY)

pipeline-all: ## Full ETL all entities
	python main.py pipeline-all
