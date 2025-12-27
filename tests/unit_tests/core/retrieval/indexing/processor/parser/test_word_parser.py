# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Word 文件解析器测试用例
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import os

from openjiuwen.core.retrieval.indexing.processor.parser.word_parser import WordParser


class TestWordParser:
    """Word 文件解析器测试"""

    def test_init(self):
        """测试初始化"""
        parser = WordParser()
        assert parser is not None

    @pytest.mark.asyncio
    async def test_parse_docx_success(self):
        """测试解析 DOCX 文件成功"""
        parser = WordParser()
        
        # 模拟 docx.Document
        mock_doc = MagicMock()
        mock_element1 = MagicMock()
        mock_element1.tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
        mock_element1.text = "Paragraph 1"
        mock_element2 = MagicMock()
        mock_element2.tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
        mock_element2.text = "Paragraph 2"
        
        mock_doc.element.body = [mock_element1, mock_element2]
        
        with patch("asyncio.to_thread") as mock_to_thread:
            # 第一次调用返回 Document 实例
            # 第二次调用返回元素列表
            mock_to_thread.side_effect = [
                mock_doc,
                [mock_element1, mock_element2],
            ]
            
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
                temp_path = f.name
            
            try:
                documents = await parser.parse(temp_path, doc_id="doc_1")
                assert len(documents) == 1
                assert documents[0].id_ == "doc_1"
                assert "Paragraph 1" in documents[0].text
                assert "Paragraph 2" in documents[0].text
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_docx_empty_document(self):
        """测试解析空文档"""
        parser = WordParser()
        
        mock_doc = MagicMock()
        mock_doc.element.body = []
        
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = [
                mock_doc,
                [],
            ]
            
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
                temp_path = f.name
            
            try:
                documents = await parser.parse(temp_path, doc_id="doc_1")
                # 空文档应该返回空列表
                assert len(documents) == 0
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_docx_file_not_found(self):
        """测试解析不存在的文件"""
        parser = WordParser()
        documents = await parser.parse("nonexistent.docx", doc_id="doc_1")
        # 应该返回空列表（因为异常被捕获）
        assert len(documents) == 0

    @pytest.mark.asyncio
    async def test_parse_docx_with_exception(self):
        """测试解析时发生异常"""
        parser = WordParser()
        
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = Exception("DOCX parsing error")
            
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
                temp_path = f.name
            
            try:
                documents = await parser.parse(temp_path, doc_id="doc_1")
                # 应该返回空列表（因为异常被捕获）
                assert len(documents) == 0
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_docx_element_paragraph(self):
        """测试解析段落元素"""
        parser = WordParser()
        
        mock_doc = MagicMock()
        mock_element = MagicMock()
        mock_element.tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
        mock_element.text = "Test paragraph"
        
        result = parser._parse_docx_element(mock_element, mock_doc)
        assert result == "Test paragraph"

    @pytest.mark.asyncio
    async def test_parse_docx_element_empty_paragraph(self):
        """测试解析空段落"""
        parser = WordParser()
        
        mock_doc = MagicMock()
        mock_table = MagicMock()
        mock_table.rows = []
        mock_table._element = MagicMock()
        mock_table._element.tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tbl"
        mock_doc.tables = [mock_table]
        
        mock_element = MagicMock()
        mock_element.tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
        mock_element.text = "   "
        
        result = parser._parse_docx_element(mock_element, mock_doc)
        assert result == ""

    @pytest.mark.asyncio
    async def test_parse_docx_element_unknown(self):
        """测试解析未知元素类型"""
        parser = WordParser()
        
        mock_doc = MagicMock()
        mock_element = MagicMock()
        mock_element.tag = "unknown:tag"
        
        result = parser._parse_docx_element(mock_element, mock_doc)
        assert result == ""

