# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Any, Dict, List, Union

from openjiuwen.core.utils.llm.messages import BaseMessage

JSONLike = Union[Dict[str, Any], List[Any]]
ChatMessage = Union[BaseMessage, Dict[str, str]]
