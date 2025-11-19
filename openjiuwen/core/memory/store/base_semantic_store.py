#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

@dataclass
class SearchHit:
    id: str  # ID of the hit vector
    distance: float  # distance from the query vector

class BaseSemanticStore(ABC):
    @abstractmethod
    def add(self, mem: List[str], memory_id: List[str], user_id: str, app_id: str,
            mem_type: str | None = None) -> None:
        pass

    @abstractmethod
    def remove(self, ids: List[str], user_id: str, app_id: str, mem_type: str | None = None) -> None:
        pass

    @abstractmethod
    def search(self, query: List[str], user_id: str, app_id: str, mem_type: str | None = None, top_k: int = 5) -> List[SearchHit]:
        pass

    @abstractmethod
    def delete_index_by_match(self, match_str: str) -> None:
        pass