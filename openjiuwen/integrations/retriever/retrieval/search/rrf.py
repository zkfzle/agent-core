# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from collections import defaultdict
from collections.abc import Hashable


def reciprocal_rank_fusion(
    rankings: list[list[Hashable]],
    *,
    k: int | float = 60,
) -> list[Hashable]:
    # https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
    key2score = defaultdict(float)

    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            key2score[key] += 1 / (rank + k)

    return sorted(key2score.keys(), key=lambda key: key2score[key], reverse=True)


def weighted_reciprocal_rank_fusion(
    rankings: list[list[Hashable]],
    weights: list[float],
    *,
    k: int | float = 60,
) -> list[Hashable]:
    if len(weights) != len(rankings):
        raise ValueError("length of weights must be aligned with rankings")

    key2score = defaultdict(float)

    for ranking, weight in zip(rankings, weights):
        for rank, key in enumerate(ranking, start=1):
            key2score[key] += weight / (rank + k)

    return sorted(key2score.keys(), key=lambda key: key2score[key], reverse=True)
