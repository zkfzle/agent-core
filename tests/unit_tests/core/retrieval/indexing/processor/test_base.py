# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
处理器抽象基类测试用例
"""
import pytest
from unittest.mock import AsyncMock

from openjiuwen.core.retrieval.indexing.processor.base import Processor


class ConcreteProcessor(Processor):
    """具体处理器实现，用于测试抽象基类"""

    async def process(self, *args, **kwargs):
        return "processed_result"


class TestProcessor:
    """处理器抽象基类测试"""

    @pytest.mark.asyncio
    async def test_process(self):
        """测试处理方法"""
        processor = ConcreteProcessor()
        result = await processor.process("test", key="value")
        assert result == "processed_result"

    def test_cannot_instantiate_abstract_class(self):
        """测试不能直接实例化抽象类"""
        with pytest.raises(TypeError):
            Processor()

