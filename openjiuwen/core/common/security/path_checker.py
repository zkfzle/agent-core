# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
import threading
import ntpath
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, List, Tuple


_WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:[\\\\/]")


@dataclass(frozen=True)
class _SensitiveEntry:
    path: str
    is_dir: bool


class PathChecker:
    _instance: Optional["PathChecker"] = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._sensitive_posix: List[_SensitiveEntry] = []
        self._sensitive_windows: List[_SensitiveEntry] = []
        self._enabled = True
        self._load_config()
        self._initialized = True

    def _load_config(self) -> None:
        self._sensitive_posix.clear()
        self._sensitive_windows.clear()

        try:
            from openjiuwen.core.common.security.user_config import UserConfig
            self._enabled = UserConfig.is_sensitive()
            sensitive_paths = UserConfig.get_sensitive_paths()
        except Exception:
            self._enabled = True
            sensitive_paths = [
                "/etc/passwd", "/etc/shadow", "/etc/hosts", "/etc/hostname",
                "/etc/ssh/", "/proc/", "/sys/", "/dev/",
                "C:\\Windows\\System32\\", "C:\\Windows\\SysWOW64\\", "C:\\Windows\\System\\"
            ]

        if not self._enabled:
            return

        for path in sensitive_paths:
            if not path or not isinstance(path, str):
                continue

            try:
                entry = self._normalize_sensitive(path.strip())
                if entry is None:
                    continue
                if self._is_windows_path(path):
                    self._sensitive_windows.append(entry)
                else:
                    self._sensitive_posix.append(entry)
            except (OSError, ValueError):
                entry = _SensitiveEntry(path=path.strip(), is_dir=path.strip().endswith(("/", "\\")))
                if self._is_windows_path(path):
                    self._sensitive_windows.append(entry)
                else:
                    self._sensitive_posix.append(entry)

    @staticmethod
    def _is_windows_path(path: str) -> bool:
        if _WINDOWS_PATH_RE.match(path) or path.startswith("\\\\"):
            return True
        return os.name == "nt"

    @staticmethod
    def _expand_path(path: str, windows: bool) -> str:
        if windows:
            return ntpath.expanduser(ntpath.expandvars(path))
        return os.path.expanduser(os.path.expandvars(path))

    def _normalize_sensitive(self, path: str) -> Optional[_SensitiveEntry]:
        if not path:
            return None
        is_dir = path.endswith(("/", "\\"))
        windows = self._is_windows_path(path)
        expanded = self._expand_path(path, windows)
        if windows:
            normalized = ntpath.normcase(ntpath.normpath(expanded))
        else:
            normalized = posixpath.normpath(expanded)
            normalized = os.path.realpath(normalized)
        normalized = normalized.rstrip("\\/") or normalized
        return _SensitiveEntry(path=normalized, is_dir=is_dir)

    def _normalize_input(self, path: Union[str, Path]) -> Tuple[str, str]:
        raw = str(path)
        windows = self._is_windows_path(raw)
        expanded = self._expand_path(raw, windows)
        if windows:
            normalized = ntpath.normcase(ntpath.normpath(expanded))
            sep = "\\"
        else:
            normalized = posixpath.normpath(expanded)
            normalized = os.path.realpath(normalized)
            sep = "/"
        normalized = normalized.rstrip("\\/") or normalized
        return normalized, sep

    def is_sensitive_path(self, path: Union[str, Path, None]) -> bool:
        if not path or not isinstance(path, (str, Path)):
            return False

        try:
            if not self._enabled:
                return False
            normalized_path, sep = self._normalize_input(path)
            sensitive_paths = self._sensitive_windows if self._is_windows_path(str(path)) else self._sensitive_posix
            for entry in sensitive_paths:
                if entry.is_dir:
                    if normalized_path == entry.path or normalized_path.startswith(entry.path + sep):
                        return True
                else:
                    if normalized_path == entry.path:
                        return True
            return False
        except (OSError, ValueError):
            return True


def is_sensitive_path(path):
    """check path if sensitive"""
    return PathChecker().is_sensitive_path(path)
