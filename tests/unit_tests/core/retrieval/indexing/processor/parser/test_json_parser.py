# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
JSON 文件解析器测试用例
"""
import pytest
from unittest.mock import patch, AsyncMock
import tempfile
import os
import json

from openjiuwen.core.retrieval.indexing.processor.parser.json_parser import JSONParser


class TestJSONParser:
    """JSON 文件解析器测试"""

    def test_init(self):
        """测试初始化"""
        parser = JSONParser()
        assert parser is not None

    @pytest.mark.asyncio
    async def test_parse_json_success(self):
        """测试解析 JSON 文件成功"""
        parser = JSONParser()
        
        # 创建临时 JSON 文件
        json_data = {
            "name": "test",
            "value": 123,
            "items": ["item1", "item2"],
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(json_data, f)
            temp_path = f.name

        try:
            documents = await parser.parse(temp_path, doc_id="doc_1")
            assert len(documents) == 1
            assert documents[0].id_ == "doc_1"
            # JSON 应该被格式化为字符串
            assert "test" in documents[0].text
            assert "123" in documents[0].text
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_json_empty_object(self):
        """测试解析空 JSON 对象"""
        parser = JSONParser()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump({}, f)
            temp_path = f.name

        try:
            documents = await parser.parse(temp_path, doc_id="doc_1")
            assert len(documents) == 1
            assert documents[0].text == "{}"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_json_array(self):
        """测试解析 JSON 数组"""
        parser = JSONParser()
        
        json_data = [1, 2, 3, "test"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(json_data, f)
            temp_path = f.name

        try:
            documents = await parser.parse(temp_path, doc_id="doc_1")
            assert len(documents) == 1
            assert "1" in documents[0].text
            assert "test" in documents[0].text
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_json_invalid_format(self):
        """测试解析无效的 JSON 格式"""
        parser = JSONParser()
        
        # 创建无效的 JSON 文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            documents = await parser.parse(temp_path, doc_id="doc_1")
            # 应该返回原始内容（因为 JSON 解析失败）
            assert len(documents) == 1
            assert "invalid json" in documents[0].text
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_json_file_not_found(self):
        """测试解析不存在的文件"""
        parser = JSONParser()
        documents = await parser.parse("nonexistent.json", doc_id="doc_1")
        # 应该返回空列表（因为异常被捕获）
        assert len(documents) == 0

    @pytest.mark.asyncio
    async def test_parse_json_with_exception(self):
        """测试解析时发生异常"""
        parser = JSONParser()
        
        with patch("aiofiles.open") as mock_open:
            mock_open.side_effect = Exception("File read error")
            
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                temp_path = f.name
            
            try:
                documents = await parser.parse(temp_path, doc_id="doc_1")
                # 应该返回空列表（因为异常被捕获）
                assert len(documents) == 0
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_json_with_unicode(self):
        """测试解析包含 Unicode 字符的 JSON"""
        parser = JSONParser()
        
        json_data = {
            "name": "测试",
            "description": "这是一个测试",
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False)
            temp_path = f.name

        try:
            documents = await parser.parse(temp_path, doc_id="doc_1")
            assert len(documents) == 1
            # 应该保留 Unicode 字符
            assert "测试" in documents[0].text
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_json_formatted_output(self):
        """测试 JSON 格式化输出"""
        parser = JSONParser()
        
        json_data = {"key": "value", "nested": {"inner": "data"}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(json_data, f)
            temp_path = f.name

        try:
            documents = await parser.parse(temp_path, doc_id="doc_1")
            assert len(documents) == 1
            # 应该被格式化为带缩进的 JSON
            parsed_text = documents[0].text
            # 验证包含缩进（格式化后的 JSON 应该有换行）
            assert "\n" in parsed_text or "  " in parsed_text
        finally:
            os.unlink(temp_path)

