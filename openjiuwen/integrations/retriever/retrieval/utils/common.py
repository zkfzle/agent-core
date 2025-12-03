# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import hashlib
from collections.abc import Callable, Hashable, Iterable
from typing import TypeVar

# Generic
T = TypeVar("T")


def deduplicate(
    data: Iterable[T],
    key: Callable[[T], Hashable] = lambda x: x,
) -> list[T]:
    exist = set()
    ret = []
    for item in data:
        val = key(item)
        if val in exist:
            continue
        exist.add(val)
        ret.append(item)
    return ret


def get_str_hash(s: str) -> str:
    hash_obj = hashlib.sha1(s.encode())
    return hash_obj.hexdigest()
