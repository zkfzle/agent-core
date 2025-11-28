#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import Optional

from openjiuwen.core.memory.store.base_semantic_store import SearchHit


def generate_idx_name(usr_id: str, app_id: str, agent_id: Optional[str] = None, mem_type: Optional[str] = None):
    """generate vector idx name"""
    if agent_id:
        if mem_type:
            return 'agent^{}^{}^{}^{}'.format(usr_id, app_id, agent_id, mem_type)
        else:
            return 'agent^{}^{}^{}^null'.format(usr_id, app_id, agent_id)
    if mem_type:
        return 'agent^{}^{}^null^{}'.format(usr_id, app_id, mem_type)
    return 'agent^{}^{}^null^null'.format(usr_id, app_id)

def parse_memory_hit_infos(hits: list[SearchHit]) -> tuple[list[str], dict[str, float]]:
    try:
        ids = [hit.id for hit in hits] if hits else []
        scores = {hit.id: hit.distance for hit in hits} if hits else {}
        return ids, scores
    except AttributeError:
        raise ValueError("SearchHit missing id attribute")
    except Exception as e:
        raise  ValueError(f"Failed to parse List[SearchHit]: {e}")
