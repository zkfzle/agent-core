import os
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from openjiuwen.core.context_engine.context_engine import ContextEngine
from openjiuwen.core.context_engine.processor.base import ContextProcessor, ContextEvent
from openjiuwen.core.context_engine.base import ModelContext
from openjiuwen.core.foundation.llm1 import (
    BaseMessage, AssistantMessage, SystemMessage, UserMessage,
    ModelRequestConfig, ModelClientConfig, Model, JsonOutputParser
)
from openjiuwen.core.context_engine.processor.context_utils import ContextUtils

os.environ.setdefault("LLM_SSL_VERIFY", "false")

DEFAULT_COMPRESSION_PROMPT: str = """
你是上下文精炼助手。
唯一任务：把下方对话历史压缩到 **不超过 500 个token** 的总结，并保留所有对未来回复有用的事实、决策、约束与关键数值。

---

📌 任务处理规则（必须严格执行）：

2. - **输出要求：**
    - 转换为自然语言描述，保留业务数据结构（如分类、差异、功能点等）
    - **保留所有与任务相关的具体信息点**（如“AI Agent将更自主地执行任务”）
    - 禁止省略任何与任务子问题对应的内容
    - **如果有工具调用，在压缩后的最终内容前加上工具调用提示词**（如"通过 get_financials 工具获得了A公司三年的财报")

---

📌 输出要求（必须严格遵守）：
- 输出合法 JSON，务必以```json```形式包裹：
```json
{
    "summary": "<精炼后的正文>"
}
```

- 所有内容必须可溯源到原文，不允许任何推断或创造。

---

messages（原始文本）：
----------------
{{messages}}
----------------

请执行：

"""


class CurrentRoundCompressorConfig(BaseModel):
    messages_threshold: int = Field(default=None, gt=0)
    """Maximum number of messages allowed in memory before offloading is triggered."""

    tokens_threshold: int = Field(default=10000, gt=0)
    """Maximum accumulated token count before offloading is triggered."""

    messages_to_keep: int = Field(default=None, gt=0)
    """Guaranteed number of most-recent messages to retain, regardless of any other threshold."""
    large_message_threshold: int = Field(default=1000, gt=0)

    customized_compression_prompt: str | None = Field(default=None)
    """User-supplied prompt for the compression/summary step; falls back to built-in prompt if None."""

    enable_kv_cache_release: bool = Field(default=True)
    # 默认False表示压缩单条超长消息，True表示压缩整块消息
    single_multi_compression: bool = Field(default=False)

    model: ModelRequestConfig | None = Field(default=None)
    """
    Reference to the model configuration, used to obtain the correct tokenizer and context-window limits. 
    If omitted, the offloader falls back to conservative defaults.
    """

    model_client: ModelClientConfig | None = Field(default=None)
    """
    Optional client-level configuration (endpoint, timeout, retry, etc.) 
    for the model used during compression/summary generation. 
    If omitted, the offloader uses the default client settings.
    """



@ContextEngine.register_processor()
class CurrentRoundCompressor(ContextProcessor):
    def __init__(self, config: CurrentRoundCompressorConfig):
        super().__init__(config)
        self._compressed_prompt = (
            config.customized_compression_prompt
            if config.customized_compression_prompt
            else DEFAULT_COMPRESSION_PROMPT
        )
        self._token_threshold = config.tokens_threshold
        self._message_num_threshold = config.messages_threshold
        self._messages_to_keep = config.messages_to_keep
        self._single_multi_config = config.single_multi_compression
        self._model_config = config.model
        self._large_message_threshold = config.large_message_threshold

        self._model = Model(
            self.config.model_client,
            self.config.model
        )

    async def on_add_messages(self,
                              context: ModelContext,
                              messages_to_add: List[BaseMessage],
                              **kwargs
                              ) -> Tuple[ContextEvent | None, List[BaseMessage]]:
        context_messages = context.get_messages() + messages_to_add
        token_counter = context.token_counter()
        last_user_idx = await self.get_compress_idx(context_messages)
        end_idx = len(context_messages) - 1
        if last_user_idx == -1:
            return None, messages_to_add
        event = ContextEvent(event_type=self.processor_type())
        if self._single_multi_config:
            compressed_context = await self.multi_compress(
                context_messages,
                last_user_idx, end_idx
            )
            if compressed_context:
                event.messages_to_modify += list(range(last_user_idx, end_idx))
                context.set_messages(compressed_context)
                return event, []
            else:
                return None, messages_to_add
        else:
            try:
                compressed_context = await self.single_compress(
                    context_messages,
                    last_user_idx, end_idx, token_counter
                )
            except Exception as e:
                raise e
            event.messages_to_modify += list(range(last_user_idx, end_idx))
            context.set_messages(compressed_context)
            return None, []


    async def trigger_add_messages(self,
                                   context: ModelContext,
                                   messages_to_add: List[BaseMessage],
                                   **kwargs
                                   ) -> bool:
        messages_num = len(context) + len(messages_to_add)
        if messages_num > self._message_num_threshold:
            return True
        if messages_num < self._messages_to_keep:
            return False
        token_counter = context.token_counter()
        tokens = 0
        if token_counter:
            context_token = token_counter.count_messages(context.get_messages())
            messages_to_add_token = token_counter.count_messages(messages_to_add)
            tokens = messages_to_add_token + context_token
        if tokens > self._token_threshold:
            return True
        return False

    @staticmethod
    async def get_compress_idx(messages: List[BaseMessage]) -> int:
        compressed_idx = -1
        for i in range(len(messages) - 1, 0, -1):
            if isinstance(messages[i], UserMessage):
                compressed_idx = i
                break
        if compressed_idx == len(messages) - 1:
            return -1

        if compressed_idx < 0:
            return -1

        return compressed_idx

    async def multi_compress(
        self,
        context_messages: List[BaseMessage],
        last_user_idx: int, end_idx: int
    ) -> Optional[list[BaseMessage]]:
        start_idx = last_user_idx + 1
        end_idx = end_idx
        if end_idx >= start_idx:
            if isinstance(context_messages[end_idx], AssistantMessage):
                if context_messages[end_idx].tool_calls:
                    end_idx = end_idx - 1
                    if end_idx < start_idx:
                        return None
        messages_to_compress = context_messages[start_idx:end_idx]
        compressed_context = await self.compress(messages_to_compress)
        if compressed_context:
            context_messages = ContextUtils.replace_messages(
                context_messages,
                [compressed_context],
                start_idx,
                end_idx
            )

        return context_messages

    async def single_compress(
        self,
        context_messages: List[BaseMessage],
        last_user_idx: int,
        end_idx: int, counter
    ) -> Optional[list[BaseMessage]]:
        start_idx = last_user_idx + 1
        end_idx = end_idx
        token_counter = counter
        for idx in range(start_idx, end_idx):
            msg = context_messages[idx]

            context_token = token_counter.count_messages([msg])
            if context_token > self._large_message_threshold:
                compressed_context = await self.compress([msg])
                if compressed_context:
                    context_messages = ContextUtils.replace_messages(
                        context_messages,
                        [compressed_context], idx, idx
                    )
            else:
                continue
        return context_messages

    async def compress(self, messages: List[BaseMessage]) -> Optional[BaseMessage]:
        processed_messages = [
            UserMessage(content=f"role:{msg.role}, content:{msg.content}")
            for msg in messages
        ]
        response = await self._model.invoke(
            [
                SystemMessage(content=self._compressed_prompt),
                *processed_messages
            ],
            output_parser=JsonOutputParser()
        )
        summary = response.parser_content
        if summary and isinstance(summary, dict):
            summary = summary.get("summary", "")
            ai_message = AssistantMessage(content=summary)
            return ai_message
        else:
            return None
