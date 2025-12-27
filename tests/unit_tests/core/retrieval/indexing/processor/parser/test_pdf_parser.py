# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
PDF 文件解析器测试用例
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import os

from openjiuwen.core.retrieval.indexing.processor.parser.pdf_parser import PDFParser


class TestPDFParser:
    """PDF 文件解析器测试"""

    def test_init(self):
        """测试初始化"""
        parser = PDFParser()
        assert parser is not None

    @pytest.mark.asyncio
    async def test_parse_pdf_success(self):
        """测试解析 PDF 文件成功"""
        parser = PDFParser()
        
        # 模拟 pdfplumber
        mock_pdf = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_pdf.pages = [mock_page1, mock_page2]
        
        with patch("pdfplumber.open") as mock_open, \
             patch("asyncio.to_thread") as mock_to_thread:
            mock_open.return_value.__enter__.return_value = mock_pdf
            mock_to_thread.return_value = "Page 1 content\nPage 2 content"
            
            # 创建临时文件路径
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                temp_path = f.name
            
            try:
                documents = await parser.parse(temp_path, doc_id="doc_1")
                assert len(documents) == 1
                assert documents[0].id_ == "doc_1"
                assert "Page 1 content" in documents[0].text
                assert "Page 2 content" in documents[0].text
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_pdf_empty_pages(self):
        """测试解析空页面的 PDF"""
        parser = PDFParser()
        
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_pdf.pages = [mock_page]
        
        with patch("pdfplumber.open") as mock_open, \
             patch("asyncio.to_thread") as mock_to_thread:
            mock_open.return_value.__enter__.return_value = mock_pdf
            mock_to_thread.return_value = ""
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                temp_path = f.name
            
            try:
                documents = await parser.parse(temp_path, doc_id="doc_1")
                # 空页面应该返回空列表
                assert len(documents) == 0
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_pdf_with_none_text(self):
        """测试解析返回 None 的页面"""
        parser = PDFParser()
        
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = None
        mock_pdf.pages = [mock_page]
        
        with patch("pdfplumber.open") as mock_open, \
             patch("asyncio.to_thread") as mock_to_thread:
            mock_open.return_value.__enter__.return_value = mock_pdf
            mock_to_thread.return_value = ""
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                temp_path = f.name
            
            try:
                documents = await parser.parse(temp_path, doc_id="doc_1")
                assert len(documents) == 0
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_pdf_file_not_found(self):
        """测试解析不存在的文件"""
        parser = PDFParser()
        documents = await parser.parse("nonexistent.pdf", doc_id="doc_1")
        # 应该返回空列表（因为异常被捕获）
        assert len(documents) == 0

    @pytest.mark.asyncio
    async def test_parse_pdf_with_exception(self):
        """测试解析时发生异常"""
        parser = PDFParser()
        
        with patch("pdfplumber.open") as mock_open:
            mock_open.side_effect = Exception("PDF parsing error")
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                temp_path = f.name
            
            try:
                documents = await parser.parse(temp_path, doc_id="doc_1")
                # 应该返回空列表（因为异常被捕获）
                assert len(documents) == 0
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_pdf_multiple_pages(self):
        """测试解析多页 PDF"""
        parser = PDFParser()
        
        mock_pdf = MagicMock()
        mock_pages = []
        for i in range(5):
            mock_page = MagicMock()
            mock_page.extract_text.return_value = f"Page {i+1} content"
            mock_pages.append(mock_page)
        mock_pdf.pages = mock_pages
        
        with patch("pdfplumber.open") as mock_open, \
             patch("asyncio.to_thread") as mock_to_thread:
            mock_open.return_value.__enter__.return_value = mock_pdf
            mock_to_thread.return_value = "\n".join([f"Page {i+1} content" for i in range(5)])
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                temp_path = f.name
            
            try:
                documents = await parser.parse(temp_path, doc_id="doc_1")
                assert len(documents) == 1
                # 验证所有页面内容都在
                for i in range(5):
                    assert f"Page {i+1} content" in documents[0].text
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

