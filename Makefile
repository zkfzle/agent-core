# Get all changed files
CHANGED_FILES := $(shell \
	git diff --name-only --diff-filter=ACM HEAD \
	| grep '\.py$$' \
)

ifndef CHANGED_FILES
$(error No changed Python files, make sure you have used git to add your changes first)
endif

# Install dependencies via uv
install:
	uv pip install ruff mypy types-requests

# Formatting check via ruff
format:
	ruff format --check $(CHANGED_FILES)

# Linting check via ruff
lint:
	ruff check --show-fixes $(CHANGED_FILES)

# Fix formatting errors via ruff
fix-format:
	ruff format $(CHANGED_FILES)

# Fix linting errors via ruff
fix-lint:
	ruff check --fix $(CHANGED_FILES)

# Use mypy for type-checking
type-check:
	mypy $(CHANGED_FILES)

# Execute all checks
check: format lint

# Execute all auto-fixes
fix: fix-lint fix-format
