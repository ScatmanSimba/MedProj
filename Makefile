# Variables
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
BLACK := $(VENV)/bin/black
FLAKE8 := $(VENV)/bin/flake8
MYPY := $(VENV)/bin/mypy
SPACY := $(VENV)/bin/spacy

# Directories
SRC_DIR := src
TEST_DIR := tests
DATA_DIR := data
CONFIG_DIR := configs
NOTEBOOK_DIR := notebooks

# Python files
PYTHON_FILES := $(shell find $(SRC_DIR) -name "*.py")
TEST_FILES := $(shell find $(TEST_DIR) -name "*.py")

# Default target
.PHONY: all
all: setup test lint

# Virtual environment setup
.PHONY: setup
setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(SPACY) download en_core_web_sm
	touch $(VENV)/bin/activate

# Testing
.PHONY: test
test: setup
	$(PYTEST) $(TEST_DIR) -v --cov=$(SRC_DIR)

# Code quality
.PHONY: lint
lint: setup
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR)
	$(MYPY) $(SRC_DIR) $(TEST_DIR)

# Format code
.PHONY: format
format: setup
	$(BLACK) $(SRC_DIR) $(TEST_DIR)

# Data processing
.PHONY: process-data
process-data: setup
	$(PYTHON) $(SRC_DIR)/data_loader.py

# Model training
.PHONY: train
train: setup process-data
	$(PYTHON) $(SRC_DIR)/train_model.py

# Generate documentation
.PHONY: docs
docs: setup
	$(PYTHON) -m pdoc --html $(SRC_DIR) --output-dir docs

# Clean up
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type d -name ".ruff_cache" -exec rm -r {} +

# Deep clean (including virtual environment)
.PHONY: deep-clean
deep-clean: clean
	rm -rf $(VENV)
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf docs

# Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  setup        - Create virtual environment and install dependencies"
	@echo "  test         - Run tests with coverage"
	@echo "  lint         - Run code quality checks (black, flake8, mypy)"
	@echo "  format       - Format code using black"
	@echo "  process-data - Process raw data"
	@echo "  train        - Train the model"
	@echo "  docs         - Generate documentation"
	@echo "  clean        - Remove Python cache files"
	@echo "  deep-clean   - Remove all generated files including virtual environment"
	@echo "  help         - Show this help message" 