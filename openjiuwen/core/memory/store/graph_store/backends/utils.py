# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

__all__ = ["batched"]

import itertools
from typing import Callable, Generator, Iterable

batched: Callable[[Iterable, int], Generator]

# itertools.batch was added in Python 3.12
if hasattr(itertools, "batched"):
    batched = getattr(itertools, "batched")
else:

    def batched(iterable: Iterable, n: int, *, strict=False) -> Generator:
        """Taken from https://docs.python.org/3/library/itertools.html#itertools.batched with modifications"""
        if n < 1:
            raise ValueError("n must be at least one")
        iterator = iter(iterable)
        batch = tuple(itertools.islice(iterator, n))
        while batch:
            if strict and len(batch) != n:
                raise ValueError("batched(): incomplete batch")
            yield batch
            batch = tuple(itertools.islice(iterator, n))
