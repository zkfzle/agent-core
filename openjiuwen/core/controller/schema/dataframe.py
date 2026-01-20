"""数据帧数据模型定义

该模块定义了数据帧（DataFrame）相关的数据模型，用于在控制器中传输不同类型的数据。

支持的数据类型：
- TextDataFrame: 文本数据
- FileDataFrame: 文件数据（支持bytes和URI两种方式）
- JsonDataFrame: JSON格式数据

DataFrame是控制器中数据传递的基本单元，用于在事件、任务输入输出等场景中传递数据。
"""
from typing import Literal, Optional, Dict, Any, Union

from pydantic import BaseModel


class BaseDataFrame(BaseModel):
    """数据帧基类
    
    定义数据帧的基本结构，支持文本、文件和JSON三种类型。
    所有具体的数据帧类型都继承自此类。
    
    Attributes:
        type: 数据帧类型，必须是"text"、"file"或"json"之一
    """
    type: Literal["text", "file", "json"]


class TextDataFrame(BaseDataFrame):
    """文本数据帧
    
    用于传输文本类型的数据。
    适用于传输纯文本内容，如用户输入、任务描述等。
    
    Attributes:
        type: 数据帧类型，固定为"text"
        text: 文本内容
    """
    type: Literal["text", "file", "json"] = "text"
    text: str


class FileDataFrame(BaseDataFrame):
    """文件数据帧
    
    用于传输文件类型的数据，支持bytes和URI两种方式。
    适用于传输文件内容，如图片、文档等。
    
    Attributes:
        type: 数据帧类型，固定为"file"
        name: 文件名
        mimeType: MIME类型，如"image/png"、"application/pdf"等
        bytes: 文件内容的字节数据（可选，与uri二选一）
        uri: 文件URI（可选，与bytes二选一）
    """
    type: Literal["text", "file", "json"] = "file"
    name: str
    mimeType: str
    bytes: Optional[bytes] = None
    uri: Optional[str] = None


class JsonDataFrame(BaseDataFrame):
    """JSON数据帧
    
    用于传输JSON格式的数据。
    适用于传输结构化数据，如配置信息、API响应等。
    
    Attributes:
        type: 数据帧类型，固定为"json"
        data: JSON数据字典
    """
    type: Literal["text", "file", "json"] = "json"
    data: Dict[str, Any]


DataFrame = Union[TextDataFrame, FileDataFrame, JsonDataFrame]




