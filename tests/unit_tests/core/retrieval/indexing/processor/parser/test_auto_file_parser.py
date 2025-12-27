# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
自动文件解析器测试用例
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import os

from openjiuwen.core.retrieval.indexing.processor.parser.auto_file_parser import (
    AutoFileParser,
    register_parser,
    _PARSER_REGISTRY,
)
from openjiuwen.core.retrieval.indexing.processor.parser.base import Parser


class TestRegisterParser:
    """解析器注册装饰器测试"""

    def test_register_parser_decorator(self):
        """测试注册解析器装饰器"""
        # 保存原始注册表
        original_registry = _PARSER_REGISTRY.copy()
        
        try:
            class TestParser(Parser):
                async def _parse(self, file_path: str):
                    return "test content"
                
                def supports(self, doc: str) -> bool:
                    return True

            # 使用装饰器注册
            decorated_parser = register_parser([".test", ".TEST"])(TestParser)
            
            # 验证注册成功
            assert ".test" in _PARSER_REGISTRY
            assert _PARSER_REGISTRY[".test"] is not None
            
            # 验证可以创建实例
            parser_instance = _PARSER_REGISTRY[".test"]()
            assert isinstance(parser_instance, TestParser)
        finally:
            # 恢复原始注册表
            _PARSER_REGISTRY.clear()
            _PARSER_REGISTRY.update(original_registry)

    def test_register_parser_multiple_extensions(self):
        """测试注册多个扩展名"""
        original_registry = _PARSER_REGISTRY.copy()
        
        try:
            class TestParser(Parser):
                async def _parse(self, file_path: str):
                    return "test content"
                
                def supports(self, doc: str) -> bool:
                    return True

            decorated_parser = register_parser([".ext1", ".ext2", ".EXT3"])(TestParser)
            
            # 验证所有扩展名都被注册（转换为小写）
            assert ".ext1" in _PARSER_REGISTRY
            assert ".ext2" in _PARSER_REGISTRY
            assert ".ext3" in _PARSER_REGISTRY
        finally:
            _PARSER_REGISTRY.clear()
            _PARSER_REGISTRY.update(original_registry)


class TestAutoFileParser:
    """自动文件解析器测试"""

    def test_init(self):
        """测试初始化"""
        parser = AutoFileParser()
        assert parser is not None

    @pytest.mark.asyncio
    async def test_parse_pdf_file(self):
        """测试解析 PDF 文件"""
        parser = AutoFileParser()
        
        # 模拟 PDF 解析
        with patch("openjiuwen.core.retrieval.indexing.processor.parser.pdf_parser.pdfplumber") as mock_pdfplumber:
            mock_pdf = MagicMock()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "PDF content"
            mock_pdf.pages = [mock_page]
            mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
            
            with patch("asyncio.to_thread") as mock_to_thread:
                mock_to_thread.return_value = "PDF content"
                
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                    temp_path = f.name
                
                try:
                    documents = await parser.parse(temp_path, doc_id="doc_1")
                    assert len(documents) == 1
                    assert documents[0].metadata["file_ext"] == ".pdf"
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_json_file(self):
        """测试解析 JSON 文件"""
        parser = AutoFileParser()
        
        import json
        json_data = {"key": "value"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(json_data, f)
            temp_path = f.name

        try:
            documents = await parser.parse(temp_path, doc_id="doc_1")
            assert len(documents) == 1
            assert documents[0].metadata["file_ext"] == ".json"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_file_not_found(self):
        """测试解析不存在的文件"""
        parser = AutoFileParser()
        
        with pytest.raises(FileNotFoundError, match="does not exist"):
            await parser.parse("nonexistent.txt", doc_id="doc_1")

    @pytest.mark.asyncio
    async def test_parse_unsupported_format(self):
        """测试解析不支持的文件格式"""
        parser = AutoFileParser()
        
        # 创建临时文件（不支持格式）
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                await parser.parse(temp_path, doc_id="doc_1")
        finally:
            os.unlink(temp_path)

    def test_supports_existing_file(self):
        """测试支持检查（文件存在）"""
        parser = AutoFileParser()
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name

        try:
            result = parser.supports(temp_path)
            assert result is True
        finally:
            os.unlink(temp_path)

    def test_supports_nonexistent_file(self):
        """测试支持检查（文件不存在）"""
        parser = AutoFileParser()
        result = parser.supports("nonexistent.txt")
        assert result is False

    def test_supports_unsupported_format(self):
        """测试支持检查（不支持格式）"""
        parser = AutoFileParser()
        
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name

        try:
            result = parser.supports(temp_path)
            assert result is False
        finally:
            os.unlink(temp_path)

    def test_register_new_parser(self):
        """测试动态注册新解析器"""
        original_registry = _PARSER_REGISTRY.copy()
        
        try:
            class CustomParser(Parser):
                async def _parse(self, file_path: str):
                    return "custom content"
                
                def supports(self, doc: str) -> bool:
                    return True

            # 动态注册
            AutoFileParser.register_new_parser(".custom", lambda: CustomParser())
            
            # 验证注册成功
            assert ".custom" in _PARSER_REGISTRY
            
            # 验证可以使用
            parser = AutoFileParser()
            with tempfile.NamedTemporaryFile(suffix='.custom', delete=False) as f:
                f.write(b"test")
                temp_path = f.name

            try:
                result = parser.supports(temp_path)
                assert result is True
            finally:
                os.unlink(temp_path)
        finally:
            _PARSER_REGISTRY.clear()
            _PARSER_REGISTRY.update(original_registry)

    def test_get_supported_formats(self):
        """测试获取支持的文件格式"""
        formats = AutoFileParser.get_supported_formats()
        assert isinstance(formats, list)
        # 应该包含常见的格式
        assert ".txt" in formats or ".pdf" in formats or ".json" in formats

    @pytest.mark.asyncio
    async def test_parse_empty_result(self):
        """测试解析返回空结果"""
        parser = AutoFileParser()
        
        # 创建一个会被解析为空内容的文件
        with patch("openjiuwen.core.retrieval.indexing.processor.parser.txt_md_parser.aiofiles") as mock_aiofiles:
            mock_file = AsyncMock()
            mock_file.read.return_value = ""
            mock_aiofiles.open.return_value.__aenter__.return_value = mock_file
            
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
                temp_path = f.name

            try:
                documents = await parser.parse(temp_path, doc_id="doc_1")
                # 空内容应该返回空列表
                assert len(documents) == 0
            finally:
                os.unlink(temp_path)
