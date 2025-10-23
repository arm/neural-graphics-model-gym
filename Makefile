# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

# Default values
USECASE := nss
TEST_TASKS := $(shell grep -oE '^test-[a-zA-Z0-9_-]+' Makefile)$
INSTALL_TASKS := $(shell grep -oE '^install-[a-zA-Z0-9_-]+' Makefile)$

# Targets
lint: blocklint # Run blocklint non-inclusive language checks and linting of all Python src files recursively
	# Prepend pylint output messages with something so that they are easily identifiable amongst the other output from the linting tools.
	@echo "Running linting of source, scripts, docs, and top-level files"
	@{ find src/ scripts/ docs/ -name "*.py" -print0; find . -maxdepth 1 -name "*.py" -print0; } | xargs -0 pylint \
  	--msg-template="pylint ERROR {path}:{line}:{column}: {msg_id}: {msg} ({symbol})"
lint-test: # Linting of all Python test files recursively
	@echo "Running linting of the test files"
	@find tests/ -name "*.py" -print0 | xargs -0 pylint --disable=invalid-name
lint-all: lint lint-test

test: test-unit test-integration test-export
	@echo "Running all tests"
test-unit: # Run unit tests for a given USECASE arg: e.g. USECASE=nss
	@echo "Running unit tests"
	python -m tests.run_tests --test-dirs tests/usecases/$(USECASE)/unit tests/core/unit tests/scripts
test-integration: # Run integration tests for a given USECASE arg: e.g. USECASE=nss
	@echo "Running integration tests"
	python -m tests.run_tests --test-dirs tests/core/integration tests/usecases/${USECASE}/integration
test-integration-fast: # Run fast integration tests for a given USECASE arg: e.g. USECASE=nss
	@echo "Running integration tests (fast mode)"
	python -m tests.run_tests --fast-test --test-dirs tests/core/integration tests/usecases/${USECASE}/integration
test-export: # Run export integration tests for a given USECASE arg: e.g. USECASE=nss
	@echo "Running export integration tests"
	python -m tests.run_tests --test-dirs tests/core/export
test-all: ${TEST_TASKS}
	@echo "Running all tests"
test-download:
	@echo "Downloading pretrained weights and datasets"
	python tests/fetch_huggingface.py
install:
	@echo "Installing package (for development run: make install-dev)"
	pip install .
install-dev:
	@echo "Installing editable package for development"
	pip install --editable .[dev]
format: # Format files
	@echo "Formatting files with black, isort and autoflake"
	autoflake --remove-unused-variables --remove-all-unused-imports --recursive --in-place .
	isort .
	black src/ scripts/ tests/
clean: # Remove temporary directories
	@echo "Removing temporary directories"
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name ".coverage*" -delete
	@rm -rf .egg* build
	@find output -mindepth 1 -maxdepth 1 ! -name '.gitignore' -exec rm -rf {} +
	@find src/ng_model_gym/usecases/nss/model/shaders -type d -name ".slangtorch_cache" -prune -exec rm -rf {} +
	@find src/ng_model_gym/usecases/nss/model/shaders -type f -name "*.lock" -delete
coverage: # Create coverage report
	@echo "Creating unit tests coverage report"
	python -m tests.run_tests --coverage --test-dirs tests/core/unit tests/core/integration \
	tests/usecases/${USECASE}/unit  tests/usecases/${USECASE}/integration tests/core/export tests/scripts
	coverage combine
	coverage report -i
	coverage json -i
	coverage html -i -d coverage_html
bandit: # Run security check
	@echo "Running security checks"
	python -m bandit --configfile pyproject.toml -r .
copyright: # Check that copyright headers exist for all required files
	@echo "Running copyright header check on all files"
	reuse lint
blocklint: # Run blocklint non-inclusive language checks
	@echo "Running blocklint to ensure no non-inclusive language"
	find . -type f \( -path "./docs/*" -o -path "./src/*" -o -path "./tests/*" -o -path "./scripts/*" \) \
	| xargs -r sh -c 'blocklint --skip-files=src/ng_model_gym/usecases/nss/model/shaders/depth_clip.slang,scripts/safetensors_generator/fsr2_methods.py "$$@"'
build-wheel: # Build Python wheel
	@echo "Building wheel and sdist"
	hatch build
list: # List all available tasks
	@echo "Listing all available tasks"
	@grep '^[^#[:space:]].*:' Makefile | grep -v '.PHONY'

.PHONY: lint ${TEST_TASKS} ${INSTALL_TASKS} test unit-test format clean coverage bandit copyright blocklint build-wheel
