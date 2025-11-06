#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
from typing import Optional, Generator


class DLTransformer:
    def __init__(self, llm, context_manager):
        self.llm = llm
        self.context_manager = context_manager

    def transform_to_mermaid(self, dl_content: str) -> str:
        pass

    def transform_to_dsl(self, dl_content: str, resource: Optional[dict] = None) -> str:
        pass
