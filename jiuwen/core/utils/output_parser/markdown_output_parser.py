import re
from typing import Any, Iterator, Optional, Union, Dict, List
from dataclasses import dataclass

from jiuwen.core.utils.output_parser.base import BaseOutputParser
from jiuwen.core.utils.llm.messages import AIMessage
from jiuwen.core.utils.llm.messages_chunk import AIMessageChunk
from jiuwen.core.common.logging import logger


class MarkdownElementType:
    """Markdown元素类型常量"""
    HEADER = "header"
    CODE_BLOCK = "code_block"
    INLINE_CODE = "inline_code"
    LINK = "link"
    IMAGE = "image"
    TABLE = "table"
    LIST = "list"
    TEXT = "text"


@dataclass
class MarkdownElement:
    """单个Markdown元素"""
    type: str  # 元素类型
    content: Dict[str, Any]  # 元素内容
    start_pos: int  # 在原文中的起始位置
    end_pos: int  # 在原文中的结束位置
    raw: str  # 原始文本


@dataclass
class MarkdownContent:
    """Markdown内容的结构化表示"""
    raw_content: str = ""
    elements: List[MarkdownElement] = None  # 按原文顺序的所有元素
    headers: List[Dict[str, str]] = None
    code_blocks: List[Dict[str, str]] = None
    links: List[Dict[str, str]] = None
    images: List[Dict[str, str]] = None
    tables: List[str] = None
    lists: List[str] = None

    def __post_init__(self):
        if self.elements is None:
            self.elements = []
        if self.headers is None:
            self.headers = []
        if self.code_blocks is None:
            self.code_blocks = []
        if self.links is None:
            self.links = []
        if self.images is None:
            self.images = []
        if self.tables is None:
            self.tables = []
        if self.lists is None:
            self.lists = []


class MarkdownOutputParser(BaseOutputParser):
    """
    解析Markdown格式内容的输出解析器。
    支持提取标题、代码块、链接、图片、表格、列表等Markdown元素。
    """

    async def parse(self, llm_output: Union[str, AIMessage]) -> Optional[MarkdownContent]:
        """
        非流式解析LLM输出，提取并返回Markdown内容结构。

        Args:
            llm_output: LLM的输出，可以是字符串或AIMessage对象。

        Returns:
            解析后的MarkdownContent对象，如果解析失败则返回None。
        """

        if isinstance(llm_output, AIMessage):
            text = llm_output.content
        elif isinstance(llm_output, str):
            text = llm_output
        else:
            logger.warning(f"Unsupported llm_output type for parse: {type(llm_output)}")
            return None

        if not text:
            return None

        try:
            markdown_content = MarkdownContent(raw_content=text)

            self._extract_all_elements(text, markdown_content)

            self._populate_categorized_lists(markdown_content)

            return markdown_content

        except Exception as e:
            logger.error(f"An unexpected error occurred during Markdown parsing: {e}\nContent: {text}")
            return None

    async def stream_parse(self, streaming_inputs: Iterator[Union[str, AIMessageChunk]]) -> Iterator[
        Optional[MarkdownContent]]:
        """
        流式解析LLM输出，逐块提取并返回Markdown内容。

        Args:
            streaming_inputs: LLM输出的迭代器，可以是字符串块或AIMessageChunk对象。

        Yields:
            解析后的MarkdownContent对象，包含当前已解析的所有Markdown元素。
        """
        buffer = ""
        last_parsed_length = 0

        for chunk in streaming_inputs:

            if isinstance(chunk, AIMessageChunk):
                if chunk.content:
                    buffer += chunk.content
            elif isinstance(chunk, str):
                buffer += chunk
            else:
                logger.warning(f"Unsupported chunk type for stream_parse: {type(chunk)}")
                continue

            if len(buffer) > last_parsed_length:
                try:
                    markdown_content = MarkdownContent(raw_content=buffer)

                    self._extract_all_elements(buffer, markdown_content)

                    self._populate_categorized_lists(markdown_content)

                    yield markdown_content
                    last_parsed_length = len(buffer)

                except Exception as e:
                    logger.error(
                        f"An unexpected error occurred during streaming Markdown parsing: {e}\nContent: {buffer}")
                    continue

        if buffer.strip():
            try:
                markdown_content = MarkdownContent(raw_content=buffer)

                self._extract_all_elements(buffer, markdown_content)

                self._populate_categorized_lists(markdown_content)

                yield markdown_content

            except Exception as e:
                logger.error(
                    f"An unexpected error occurred during final streaming Markdown parsing: {e}\nContent: {buffer}")

    def _extract_all_elements(self, text: str, markdown_content: MarkdownContent):
        """提取所有Markdown元素并按位置排序"""
        elements = []

        # 提取标题
        for match in re.finditer(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            elements.append(MarkdownElement(
                type=MarkdownElementType.HEADER,
                content={"level": str(level), "title": title},
                start_pos=match.start(),
                end_pos=match.end(),
                raw=match.group(0)
            ))

        # 提取代码块
        for match in re.finditer(r'```(\w*)\n(.*?)\n```', text, re.DOTALL):
            language = match.group(1) or "text"
            code = match.group(2)
            elements.append(MarkdownElement(
                type=MarkdownElementType.CODE_BLOCK,
                content={"language": language, "code": code},
                start_pos=match.start(),
                end_pos=match.end(),
                raw=match.group(0)
            ))

        # 提取内联代码
        for match in re.finditer(r'`([^`\n]+)`', text):
            elements.append(MarkdownElement(
                type=MarkdownElementType.INLINE_CODE,
                content={"code": match.group(1)},
                start_pos=match.start(),
                end_pos=match.end(),
                raw=match.group(0)
            ))

        # 提取图片
        for match in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)', text):
            alt_text = match.group(1)
            url = match.group(2)
            elements.append(MarkdownElement(
                type=MarkdownElementType.IMAGE,
                content={"alt": alt_text, "url": url},
                start_pos=match.start(),
                end_pos=match.end(),
                raw=match.group(0)
            ))

        # 提取链接
        for match in re.finditer(r'(?<!\!)\[([^\]]+)\]\(([^)]+)\)', text):
            text_part = match.group(1)
            url = match.group(2)
            elements.append(MarkdownElement(
                type=MarkdownElementType.LINK,
                content={"text": text_part, "url": url},
                start_pos=match.start(),
                end_pos=match.end(),
                raw=match.group(0)
            ))

        # 提取表格和列表（这些是多行元素，需要特殊处理）
        self._extract_multiline_elements(text, elements)

        # 按位置排序
        elements.sort(key=lambda x: x.start_pos)
        markdown_content.elements = elements

    def _extract_multiline_elements(self, text: str, elements: List[MarkdownElement]):
        """提取多行元素（表格和列表）"""
        lines = text.split('\n')
        current_pos = 0

        # 处理表格和列表
        table_lines = []
        list_lines = []
        table_start_pos = -1
        list_start_pos = -1

        for i, line in enumerate(lines):
            line_start_pos = current_pos
            line_end_pos = current_pos + len(line)
            current_pos = line_end_pos + 1

            # 检查表格
            if '|' in line.strip() and line.strip():
                if not table_lines:
                    table_start_pos = line_start_pos
                table_lines.append(line)
            else:
                if table_lines:
                    # 表格结束
                    table_content = '\n'.join(table_lines)
                    elements.append(MarkdownElement(
                        type=MarkdownElementType.TABLE,
                        content={"table": table_content},
                        start_pos=table_start_pos,
                        end_pos=line_start_pos - 1,
                        raw=table_content
                    ))
                    table_lines = []

            # 检查列表
            if re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
                if not list_lines:
                    list_start_pos = line_start_pos
                list_lines.append(line)
            elif re.match(r'^\s*$', line) and list_lines:
                list_lines.append(line)
            else:
                if list_lines:
                    # 列表结束
                    list_content = '\n'.join(list_lines).strip()
                    if list_content:
                        elements.append(MarkdownElement(
                            type=MarkdownElementType.LIST,
                            content={"list": list_content},
                            start_pos=list_start_pos,
                            end_pos=line_start_pos - 1,
                            raw=list_content
                        ))
                    list_lines = []

        # 处理文档末尾的表格和列表
        if table_lines:
            table_content = '\n'.join(table_lines)
            elements.append(MarkdownElement(
                type=MarkdownElementType.TABLE,
                content={"table": table_content},
                start_pos=table_start_pos,
                end_pos=len(text),
                raw=table_content
            ))

        if list_lines:
            list_content = '\n'.join(list_lines).strip()
            if list_content:
                elements.append(MarkdownElement(
                    type=MarkdownElementType.LIST,
                    content={"list": list_content},
                    start_pos=list_start_pos,
                    end_pos=len(text),
                    raw=list_content
                ))

    def _populate_categorized_lists(self, markdown_content: MarkdownContent):
        """填充分类列表"""
        for element in markdown_content.elements:
            if element.type == MarkdownElementType.HEADER:
                markdown_content.headers.append({
                    "level": element.content["level"],
                    "title": element.content["title"],
                    "raw": element.raw
                })
            elif element.type == MarkdownElementType.CODE_BLOCK:
                markdown_content.code_blocks.append({
                    "language": element.content["language"],
                    "code": element.content["code"],
                    "raw": element.raw
                })
            elif element.type == MarkdownElementType.INLINE_CODE:
                markdown_content.code_blocks.append({
                    "language": "inline",
                    "code": element.content["code"],
                    "raw": element.raw
                })
            elif element.type == MarkdownElementType.LINK:
                markdown_content.links.append({
                    "text": element.content["text"],
                    "url": element.content["url"],
                    "raw": element.raw
                })
            elif element.type == MarkdownElementType.IMAGE:
                markdown_content.images.append({
                    "alt": element.content["alt"],
                    "url": element.content["url"],
                    "raw": element.raw
                })
            elif element.type == MarkdownElementType.TABLE:
                markdown_content.tables.append(element.content["table"])
            elif element.type == MarkdownElementType.LIST:
                markdown_content.lists.append(element.content["list"])

    def _extract_headers(self, text: str) -> List[Dict[str, str]]:
        """提取Markdown标题"""
        headers = []
        # 匹配 # ## ### 等标题
        pattern = r'^(#{1,6})\s+(.+)$'
        for match in re.finditer(pattern, text, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            headers.append({
                "level": str(level),
                "title": title,
                "raw": match.group(0)
            })
        return headers

    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """提取代码块"""
        code_blocks = []
        # 匹配 ```language\ncode\n``` 格式的代码块
        pattern = r'```(\w*)\n(.*?)\n```'
        for match in re.finditer(pattern, text, re.DOTALL):
            language = match.group(1) or "text"
            code = match.group(2)
            code_blocks.append({
                "language": language,
                "code": code,
                "raw": match.group(0)
            })

        # 匹配单行代码 `code`
        inline_pattern = r'`([^`\n]+)`'
        for match in re.finditer(inline_pattern, text):
            code_blocks.append({
                "language": "inline",
                "code": match.group(1),
                "raw": match.group(0)
            })

        return code_blocks

    def _extract_links(self, text: str) -> List[Dict[str, str]]:
        """提取链接"""
        links = []
        # 匹配 [text](url) 格式的链接
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(pattern, text):
            text_part = match.group(1)
            url = match.group(2)
            links.append({
                "text": text_part,
                "url": url,
                "raw": match.group(0)
            })
        return links

    def _extract_images(self, text: str) -> List[Dict[str, str]]:
        """提取图片"""
        images = []
        # 匹配 ![alt](url) 格式的图片
        pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        for match in re.finditer(pattern, text):
            alt_text = match.group(1)
            url = match.group(2)
            images.append({
                "alt": alt_text,
                "url": url,
                "raw": match.group(0)
            })
        return images

    def _extract_tables(self, text: str) -> List[str]:
        """提取表格"""
        tables = []
        # 匹配Markdown表格
        lines = text.split('\n')
        table_lines = []
        in_table = False

        for line in lines:
            # 检查是否是表格行（包含 | 分隔符）
            if '|' in line.strip() and line.strip():
                table_lines.append(line)
                in_table = True
            else:
                if in_table and table_lines:
                    tables.append('\n'.join(table_lines))
                    table_lines = []
                in_table = False

        if table_lines:
            tables.append('\n'.join(table_lines))

        return tables

    def _extract_lists(self, text: str) -> List[str]:
        """提取列表"""
        lists = []
        lines = text.split('\n')
        list_lines = []
        in_list = False

        for line in lines:
            # 检查是否是列表项（- * + 或数字.）
            if re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
                list_lines.append(line)
                in_list = True
            elif in_list and line.strip() == '':
                list_lines.append(line)
            else:
                if in_list and list_lines:
                    lists.append('\n'.join(list_lines).strip())
                    list_lines = []
                in_list = False

        if list_lines:
            lists.append('\n'.join(list_lines).strip())

        return lists
