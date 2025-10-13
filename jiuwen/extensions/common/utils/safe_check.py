import os
from pathlib import Path

SENSITIVE_PATHS = [
    '/etc', '/bin', '/usr/bin', '/usr/sbin', '/boot', '/root'
]


def is_valid_path(path):
    try:
        p = Path(path)
        return p.exists() and p.is_absolute()
    except Exception:
        return False


def is_sensitive_path(path):
    abs_path = os.path.abspath(path)
    for sensitive in SENSITIVE_PATHS:
        if abs_path.startswith(os.path.abspath(sensitive)):
            return True
    return False


def is_safe_path(path):
    return is_valid_path(path) and not is_sensitive_path(path)
