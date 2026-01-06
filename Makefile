# Get all staged .py files
CHANGED_FILES := $(shell \
	git diff --name-only --cached --diff-filter=ACM \
	| grep -E '\.pyi?$$' || true \
)

# This target fails if there are no staged Python files
has-staged-changes:
	@if [ -z "$(CHANGED_FILES)" ]; then \
		echo "No Python files staged for change."; \
		echo "Make sure you have used 'git add' first."; \
		exit 1; \
	fi

# Install dependencies via uv
install:
	uv pip install ruff mypy types-requests

# Formatting check via ruff
format: has-staged-changes
	@ruff format --check $(CHANGED_FILES)

# Linting check via ruff
lint: has-staged-changes
	@ruff check --show-fixes $(CHANGED_FILES)

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
check: format lint

# Execute all auto-fixes
fix: fix-lint fix-format

# All targets
.PHONY: has-staged-changes install format lint fix-format fix-lint type-check check fix
