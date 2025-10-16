#!/usr/bin/python3.11
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from __future__ import annotations

import os
import threading
import configparser
from pathlib import Path
from typing import Optional, Dict, Any

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.logging import logger


class UserConfig:
    _instance: Optional["UserConfig"] = None
    _lock = threading.Lock()

    DEFAULT: Dict[str, Any] = {"settings": {"is_sensitive": True}}

    def __init__(self, config_path: Optional[Path] = None):
        self._cfg = configparser.ConfigParser()
        self._cfg.read_dict(self.DEFAULT)
        if config_path and config_path.is_file():
            try:
                self._cfg.read(config_path, encoding="utf-8")
            except Exception:
                logger.error(f"Failed to read config file.")
        self.is_sensitive: bool = self._cfg.getboolean("settings", "is_sensitive")

    @classmethod
    def set_config_path(cls, path: Path) -> None:
        """set config file path"""
        if cls._instance is not None:
            raise JiuWenBaseException(183101, "Config already initialized.")
        cls._user_path = cls._resolve_and_check(path)

    @classmethod
    def get_config(cls) -> "UserConfig":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    user_path = getattr(cls, "_user_path", None)
                    cls._instance = cls(user_path)
        return cls._instance

    @classmethod
    def is_sensitive(cls) -> bool:
        """直接返回当前单例的 is_sensitive 值（延迟初始化）"""
        return cls.get_config().is_sensitive

    @staticmethod
    def _resolve_and_check(path: Path) -> Path:
        path = Path(os.path.expandvars(path.expanduser())).resolve()

        root = Path.cwd()
        try:
            path.relative_to(root)
        except ValueError:
            raise JiuWenBaseException(183100, "Config file must reside inside root")

        return path
