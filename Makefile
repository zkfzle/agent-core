# Define directories to check in SRC_DIRS
SRC_DIRS = openjiuwen

# Install dependencies via uv
install:
	uv pip install ruff mypy types-requests

# Formatting check via ruff
format:
	ruff format --check $(SRC_DIRS)

# Linting check via ruff
lint:
	ruff check --show-fixes $(SRC_DIRS)

# Fix formatting errors via ruff
fix-format:
	ruff format $(SRC_DIRS)

# Fix linting errors via ruff
fix-lint:
	ruff check --fix $(SRC_DIRS)

# Use mypy for type-checking
type-check:
	mypy $(SRC_DIRS)

# Execute all checks
check: format lint

# Execute all auto-fixes
fix: fix-lint fix-format
