# All targets
.PHONY: help has-staged-changes install test format lint pylint fix-format fix-lint type-check check fix

# Default target
.DEFAULT_GOAL := help

# Get all staged .py or .pyi Python files
CHANGED_FILES := $(shell \
	git diff --name-only --cached --diff-filter=ACMR \
	| grep -E '\.pyi?$$' || true \
)

# Show help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "  help       - Show this help message"
	@echo "  install    - Install dependencies via uv (ruff, pylint, mypy, types-requests)"
	@echo "  test       - Execute pytest"
	@echo "  format     - Check formatting of staged Python files via ruff"
	@echo "  lint       - Check linting of staged Python files via ruff"
	@echo "  pylint     - Check linting of staged Python files via pylint (more comprehensive)"
	@echo "  fix-format - Auto-fix formatting errors in staged Python files via ruff"
	@echo "  fix-lint   - Auto-fix linting errors in staged Python files via ruff"
	@echo "  type-check - Type-check staged Python files via mypy"
	@echo "  check      - Run all checks (format, lint, pylint)"
	@echo "  fix        - Run all auto-fixes (fix-lint, fix-format)"
	@echo ""
	@echo "Note: Most targets require staged Python files (use 'git add' first)"

# Install dependencies via uv
install:
	uv pip install "ruff>=0.11.2" "pylint>=3.0.0" "mypy>=1.12.0" "types-requests"

# Execute pytest
test:
	@pytest

# Sanity check - this target fails if there are no staged Python files
has-staged-changes:
	@if [ -z "$(CHANGED_FILES)" ]; then \
		echo "No Python files staged for change."; \
		echo "Make sure you have used 'git add' first."; \
		exit 1; \
	fi

# Formatting check via ruff
format: has-staged-changes
	@ruff format --check $(CHANGED_FILES)

# Linting check via ruff
lint: has-staged-changes
	@ruff check --show-fixes $(CHANGED_FILES)

# Linting check via pylint (more comprehensive than ruff check)
pylint: has-staged-changes
	@pylint $(CHANGED_FILES)

# Fix formatting errors via ruff
fix-format: has-staged-changes
	@ruff format $(CHANGED_FILES)

# Fix linting errors via ruff
fix-lint: has-staged-changes
	@ruff check --fix $(CHANGED_FILES)

# Use mypy for type-checking
type-check: has-staged-changes
	@mypy $(CHANGED_FILES)

# Execute all checks
check: format lint pylint

# Execute all auto-fixes
fix: fix-lint fix-format
