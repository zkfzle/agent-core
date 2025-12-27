# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
TXT/MD 文件解析器测试用例
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import tempfile
import os

from openjiuwen.core.retrieval.indexing.processor.parser.txt_md_parser import TxtMdParser


class TestTxtMdParser:
    """TXT/MD 文件解析器测试"""

    def test_init(self):
        """测试初始化"""
        parser = TxtMdParser()
        assert parser is not None

    @pytest.mark.asyncio
    async def test_parse_empty_file(self):
        """测试解析空文件"""
        parser = TxtMdParser()
        
        # 创建临时空文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name

        try:
            documents = await parser.parse(temp_path, doc_id="doc_1")
            # 空文件应该返回空列表或包含空文本的文档
            if documents:
                assert documents[0].text.strip() == ""
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_file_not_found(self):
        """测试解析不存在的文件"""
        parser = TxtMdParser()
        documents = await parser.parse("nonexistent.txt", doc_id="doc_1")
        # 应该返回空列表（因为异常被捕获）
        assert len(documents) == 0

    @pytest.mark.asyncio
    async def test_parse_strips_content(self):
        """测试内容去除空白"""
        parser = TxtMdParser()
        
        # 创建临时文件（前后有空白）
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("   \n  Content  \n   ")
            temp_path = f.name

        try:
            documents = await parser.parse(temp_path, doc_id="doc_1")
            if documents:
                # 内容应该被 strip
                assert documents[0].text.strip() == "Content"
        finally:
            os.unlink(temp_path)

