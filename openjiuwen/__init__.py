# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from importlib.metadata import PackageNotFoundError, version

try:
    # Wheel installation should include package metadata
    __version__ = version("openjiuwen")
except PackageNotFoundError:
    # Only developer working with source code directly may not have package metadata
    import tomllib
    from pathlib import Path

    __version__ = "unknown"
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    if pyproject.exists():
        project_metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        version_parsed = project_metadata.get("project", {}).get("version")
        if version_parsed:
            __version__ = version_parsed

__all__ = ["__version__"]
