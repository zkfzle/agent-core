# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
__all__ = ["ensure_valid_language", "format_relation_definitions", "get_formatting_kwargs"]

from . import cn, en
from .base import ensure_valid_language, format_relation_definitions, get_formatting_kwargs

cn.register_language()
en.register_language()
