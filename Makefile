# SRC_DIRS中定义检查目录
SRC_DIRS = openjiuwen

# 使用uv安装依赖
install:
	uv pip install ruff mypy types-requests

# 使用ruff检查格式
format:
	ruff format --check $(SRC_DIRS)

# 使用ruff检查代码
lint:
	ruff check --show-fixes $(SRC_DIRS)

# 使用ruff修复格式
fix-format:
	ruff format $(SRC_DIRS)

# 使用ruff修复代码
fix-lint:
	ruff check --fix $(SRC_DIRS)

# 使用mypy对代码进行静态类型分析
type-check:
	mypy $(SRC_DIRS)

# 执行所有检查
check: format lint

# 执行所有自动修复
fix: fix-lint fix-format
