#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from abc import abstractmethod, ABC
from typing import Dict, Any


class Serializable(ABC):
    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def deserialize(self, data: Dict[str, Any]):
        pass