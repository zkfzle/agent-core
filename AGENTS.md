# OpenJiuwen Agent Core - Developer Guide

This file serves as a definitive guide for AI agents and developers working on the `openJiuwen/agent-core` repository.

## 1. Environment & Build

The project uses `uv` for dependency management and Python 3.11+ as the runtime.

### Key Commands

| Action | Command | Notes |
|--------|---------|-------|
| **Install** | `uv sync` | Installs dependencies including dev groups |
| **Run Tests** | `uv run pytest` | Runs all tests |
| **Single Test** | `uv run pytest tests/path/to/test_file.py` | Run specific test file |
| **Lint** | `uv run ruff check .` | Checks for linting errors |
| **Format** | `uv run ruff format .` | Auto-formats code |
| **Type Check** | `uv run mypy .` | Runs static type checking |

> **Note for Agents**: Always prefix python commands with `uv run` to ensure execution in the correct virtual environment.

## 2. Code Style & Conventions

### Formatting
- **Line Length**: 120 characters (enforced by `ruff`).
- **Indentation**: 4 spaces.
- **Quotes**: Double quotes `"` preferred over single quotes `'` for strings.
- **Encoding**: File headers must include `# coding: utf-8` and Copyright.

### Imports
Order of imports:
1. Standard Library (e.g., `os`, `sys`, `typing`)
2. Third-party Libraries (e.g., `sqlalchemy`, `pydantic`, `openai`)
3. Local Application (`openjiuwen.*`)

```python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import asyncio
from typing import List, Optional

from sqlalchemy import text
from pydantic import BaseModel

from openjiuwen.core.common.logging import logger
```

### Typing
- **Strictness**: Moderate. `mypy` is configured with `check_untyped_defs = true` but `disallow_untyped_defs = false`.
- **Style**: Use modern syntax `str | int` where possible, but `List`, `Dict`, `Optional` from `typing` are still prevalent in the codebase.
- **Pydantic**: Heavily used for configuration and data models. Use `Field(..., description="...")` for fields.

### Naming
- **Classes**: `CamelCase` (e.g., `PGVectorStore`, `WorkflowAgent`)
- **Functions/Methods**: `snake_case` (e.g., `create_client`, `add_documents`)
- **Variables**: `snake_case`
- **Constants**: `UPPER_CASE`
- **Private Members**: Prefix with `_` (e.g., `_engine`, `_ensure_extension`)

### Asynchronous Programming
- The core is designed to be **async-first**.
- Use `async/await` for all I/O bound operations (DB, Network, LLM calls).
- Use `aiohttp`, `asyncpg`, `aiofiles`.
- Avoid blocking calls in async functions; use `run_in_executor` if necessary.

## 3. Architecture & Patterns

### Error Handling
- Use the unified exception handling system.
- Raise errors using `build_error` with a specific `StatusCode`.

```python
from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error

raise build_error(
    StatusCode.RETRIEVAL_KB_DATABASE_CONFIG_INVALID,
    error_msg="Invalid configuration provided"
)
```

### Logging
- Use the central logger instance.
- Do not use `print()` statements (enforced by linter).

```python
from openjiuwen.core.common.logging import logger

logger.info("Operation completed successfully")
logger.warning("Something unexpected happened")
```

### Vector Stores
- Inherit from `openjiuwen.core.retrieval.vector_store.base.VectorStore` or specific backend like `PGVectorStore`.
- Implement all abstract methods: `add`, `search`, `delete`, `create_client`, etc.
- Configs must be defined in `openjiuwen/core/retrieval/common/config.py`.
- For SQL-based stores, prefer **Raw SQL** for DDL/DML to ensure compatibility with specific dialects (e.g., openGauss `datavec`).

## 4. Testing

- **Framework**: `pytest` with `pytest-asyncio`.
- **Structure**: Tests are located in `tests/`. Unit tests mirror the source structure.
- **Fixtures**: Use `pytest` fixtures for setup/teardown.
- **Mocking**: Use `unittest.mock` (`MagicMock`, `AsyncMock`, `patch`) to mock external calls (LLM APIs, DB connections).
- **Database Tests**: Since CI environments might lack specific DBs (like openGauss), use strict SQL string verification with mocks.

## 5. Agent Workflow Rules

1.  **Plan First**: Before writing code, analyze the task and create a plan.
2.  **Verify Environment**: Ensure `uv` is installed and environment is synced.
3.  **Read Before Write**: Always read existing files to match style and import patterns.
4.  **Test Driven**: When fixing bugs or adding features, try to write a reproduction test case first.
5.  **No "Print" Debugging**: Use logging or return values.
