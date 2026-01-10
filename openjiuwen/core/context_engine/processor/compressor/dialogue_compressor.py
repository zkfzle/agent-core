#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field

from openjiuwen.core.context_engine.context_engine import ContextEngine
from openjiuwen.core.context_engine.processor.base import ContextProcessor
from openjiuwen.core.context_engine.base import ModelContext, ContextWindow
from openjiuwen.core.foundation.llm1 import (
    BaseMessage, AssistantMessage, UserMessage, ToolMessage, SystemMessage,
    ModelConfig, ModelClientConfig, Model, ToolCall, JsonOutputParser
)
from openjiuwen.core.context_engine.processor.context_utils import ContextUtils

os.environ.setdefault("LLM_SSL_VERIFY", "false")

DEFAULT_COMPRESSION_PROMPT: str = """
You are a "tool-call compressor that relies solely on the original text". You have no knowledge base, cannot use common-sense reasoning, and cannot infer or complete; you can only process the given text.

Your task: Extract and compress the shortest information segment that fully answers the user's task requirements from the tool calls and tool responses.

Rules:
- Retain only task-relevant specific information points; delete all filler, examples, or duplicates.
- Preserve business data structures (categories, differences, feature points) in natural language.
- Do not omit any sub-question content.
- Prefix the compressed content with: "Through <tool_name> tool, obtained: <compressed_text>".

Output valid JSON:
```json
{
    "summary": "<compressed_text>"
}
"""


class DialogueCompressorConfig(BaseModel):
    """
    Configuration for the MessageOffloader ContextProcessor.

    The offloader keeps the conversation history within safe memory/token limits
    by trimming or offloading messages once the configured thresholds are exceeded.
    Rules are evaluated in the following order:

    1. messages_to_keep: the most recent N messages are always retained.
    2. messages_threshold: when total message count exceeds this value offloading
       is triggered.
    3. tokens_threshold: when accumulated token count exceeds this value
       offloading is triggered.

    Only messages whose role appears in `offload_message_type` and whose token
    length is greater than `large_message_threshold` are eligible for offloading.
    The last user-assistant round can be preserved independently of the above
    rules by setting `keep_last_round=True`.
    """

    messages_threshold: int = Field(default=None, gt=0)
    """Maximum number of messages allowed in memory before offloading is triggered."""

    tokens_threshold: int = Field(default=10000, gt=0)
    """Maximum accumulated token count before offloading is triggered."""

    messages_to_keep: int = Field(default=None, gt=0)
    """Guaranteed number of most-recent messages to retain, regardless of any other threshold."""

    keep_last_round: bool = Field(default=True)
    """If True, the most recent user-assistant round is always preserved even if it would otherwise be offloaded."""

    customized_compression_prompt: str | None = Field(default=None)
    """User-supplied prompt for the compression/summary step; falls back to built-in prompt if None."""

    compression_token_limit: int = Field(default=2000, gt=0)
    """Max tokens allowed in the compressed summary; shorter summaries are preferred when possible."""

    enable_kv_cache_release: bool = Field(default=True)
    """Whether to instruct the inference backend to release KV-cache entries corresponding to offloaded messages, reducing GPU memory at the cost of slower re-generation if those messages are needed again."""

    model: ModelConfig | None = Field(default=None)
    """Reference to the model configuration, used to obtain the correct tokenizer and context-window limits.  If omitted, the offloader falls back to conservative defaults."""

    model_client: ModelClientConfig | None = Field(default=None)
    """Optional client-level configuration (endpoint, timeout, retry, etc.) for the model used during compression/summary generation.  If omitted, the offloader uses the default client settings."""


@ContextEngine.register_processor()
class DialogueCompressor(ContextProcessor):
    def __init__(self, config: DialogueCompressorConfig):
        super().__init__(config)
        self._compressed_prompt = (
            config.customized_compression_prompt
            if config.customized_compression_prompt
            else DEFAULT_COMPRESSION_PROMPT
        )
        self._token_threshold = config.tokens_threshold
        self._message_num_threshold = config.messages_threshold
        self._messages_to_keep = config.messages_to_keep

        self._model_config = config.model

        self._model = Model(
            self.config.model_client,
            self.config.model
        )


    async def on_add_messages(self,
                              context: ModelContext,
                              messages_to_add: List[BaseMessage],
                              **kwargs
                              ) -> List[BaseMessage]:

        context_messages = context.get_messages() + messages_to_add
        compressed_idx = await self.get_compress_idx(context_messages)
        if compressed_idx == -1:
            return messages_to_add

        msg_pairs = await self.get_compress_pairs(context_messages[:compressed_idx])
        if len(msg_pairs) == 0:
            return messages_to_add

        for msg_pair in msg_pairs[::-1]:
            start_idx = msg_pair[0] + 1
            end_idx = msg_pair[1]
            dialogues = []
            for i in range(start_idx, end_idx+1):
                dialogues.append(context_messages[i])
            compressed_context = await self.compress(dialogues)
            if compressed_context:
                context_messages = ContextUtils.replace_messages(
                    context_messages,
                    [compressed_context],
                    start_idx,
                    end_idx
                )

        context.set_messages(context_messages)
        return []

    async def on_get_context_window(self,
                                    context: ModelContext,
                                    context_window: ContextWindow,
                                    **kwargs
                                    ) -> ContextWindow:
        return context_window

    async def trigger_add_messages(self,
                                   context: ModelContext,
                                   messages_to_add: List[BaseMessage],
                                   **kwargs
                                   ) -> bool:
        messages_num = len(context) + len(messages_to_add)
        if self._message_num_threshold is not None and messages_num > self._message_num_threshold:
            return True
        if self._messages_to_keep is not None and messages_num < self._messages_to_keep:
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

    async def trigger_get_context_window(self,
                                         context: ModelContext,
                                         context_window: ContextWindow,
                                         **kwargs
                                         ) -> bool:
        return False

    async def get_compress_idx(self, messages: List[BaseMessage]) -> int:
        last_ai_msg_index = None
        if self.config.keep_last_round:
            last_ai_msg_index = ContextUtils.find_last_ai_message_without_tool_call(messages)
        keep_index = (
            len(messages)
            if not self.config.messages_to_keep
            else len(messages) - self.config.messages_to_keep
        )
        compressed_idx = (
            keep_index
            if last_ai_msg_index is None
            else min(last_ai_msg_index, keep_index)
        )

        return compressed_idx

    async def get_compress_pairs(self, messages: List[BaseMessage]) -> List[Tuple[int, int]]:
        current_user = -1
        result = []
        for i in range(len(messages)):
            if isinstance(messages[i], UserMessage):
                current_user = i
            elif isinstance(messages[i], AssistantMessage) and not messages[i].tool_calls and current_user != -1:
                if i - current_user > 1:
                    result.append((current_user, i))
                    current_user = -1
            else:
                continue

        return result

    async def compress(self, messages: List[BaseMessage]) -> Optional[BaseMessage]:
        messages = [SystemMessage(content=self._compressed_prompt)] + messages
        response = await self._model.ainvoke(messages, output_parser=JsonOutputParser())
        summary = response.parser_content
        if summary:
            summary = summary.get("summary", "")
            ai_message = AssistantMessage(content="[RND-SUMMARY]" + summary)
            return ai_message
        else:
            return None

    def load_state(self, state: Dict[str, Any]) -> None:
        pass

    def save_state(self) -> Dict[str, Any]:
        pass

if __name__ == "__main__":
    import asyncio
    from openjiuwen.core.context_engine.token.tiktoken_counter import TiktokenCounter
    async def run():
        ce = ContextEngine()
        context = await ce.create_context(
            "context",
            processors=[
                (
                    "DialogueCompressor",
                    DialogueCompressorConfig(
                        # messages_threshold=10,
                        tokens_threshold=200,
                        # messages_to_keep=10,
                        keep_last_round=False,
                        model=ModelConfig(
                            model="qwen2.5-72b-instruct"
                        ),
                        model_client=ModelClientConfig(
                            client_id="123",
                            client_type="OpenAI",
                            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                            api_key="sk-f9410e0700c94022a16b78341e860c45",
                            verify_ssl=False
                        )
                    )
                ),
            ],
            token_counter=TiktokenCounter()
        )

        tool_call = ToolCall(
            id="call_001",  # 可选：自定义唯一ID，也可以传 None
            type="function",  # 工具类型，通常固定为 "function"
            name="get_financials",  # 工具函数名（根层级，不是function嵌套）
            arguments=json.dumps({  # 参数必须是 JSON 字符串（关键！）
                "company": "A公司",
                "years": [2023, 2024, 2025],
                "report_type": "income_statement"
            }),
            index=0  # 可选：工具调用的索引，多工具时用
        )

        messages_batch = [
            UserMessage(
                content="我想这周末去京都玩，麻烦帮我查一下京都天气、把500美元换成日元、找今天东京飞京都的最便宜航班，再给个旅行小建议，谢谢！"),

            AssistantMessage(
                content="检测到工具调用",
                tool_calls=[
                    ToolCall(id="call_fy_1", type="function", name="get_weather", arguments='{"city": "京都"}'),
                    ToolCall(id="call_fx_1", type="function", name="fx_convert",
                             arguments='{"amount": 500, "from": "USD", "to": "JPY"}'),
                    ToolCall(id="call_fl_1", type="function", name="find_flight",
                             arguments='{"from": "东京", "to": "京都", "date": "today", "sort": "price"}')
                ]
            ),

            ToolMessage(tool_call_id="call_fy_1",
                        content='{"city": "京都", "temp": 22, "unit": "°C", "condition": "晴朗", "humidity": 55, "wind": 8, "forecast": "本周末持续晴好，白天温暖、早晚略凉，紫外线指数高，适宜出行但需注意防晒", "hourly": [{"time": "09:00", "temp": 18, "condition": "晴"}, {"time": "12:00", "temp": 22, "condition": "晴"}, {"time": "15:00", "temp": 24, "condition": "晴"}, {"time": "18:00", "temp": 20, "condition": "晴"}, {"time": "21:00", "temp": 17, "condition": "晴"}], "air_quality": "良好", "uv_index": 7, "suggestion": "涂抹 SPF30+ 防晒霜，佩戴帽子或墨镜"}'),

            ToolMessage(tool_call_id="call_fx_1",
                        content='{"amount": 500, "from": "USD", "to": "JPY", "rate": 152.3, "result": 76150, "currency": "JPY", "fee": 0, "location": "成田机场T1兑换点实时牌价", "timestamp": "2026-01-07T14:25:00+09:00", "daily_limit": 10000, "weekly_limit": 50000, "promotion": "单笔≥400 USD 免手续费，赠送京都地铁一日券折扣二维码（5% off）", "note": "需出示护照，支持现金/银行卡/支付宝，营业 06:00-22:00"}'),

            ToolMessage(tool_call_id="call_fl_1",
                        content='{"from": "东京羽田", "to": "京都伊丹", "date": "2026-01-07", "price": 12500, "currency": "JPY", "flight": "JL2201", "departure": "07:10", "arrival": "08:05", "duration": "55min", "baggage": "20kg免费", "seat": "剩余充足", "aircraft": "Boeing 737-800", "terminal": "羽田 T2 → 伊丹 T1", "check_in": "在线/自助 06:10 开放", "gate": "待定", "meal": "免费茶水+小点心", "wifi": "机内 Wi-Fi 免费", "pet": "可托运", "extra": "绿色飞行积分 300 里程", "return_discount": "同程返程票 9 折", "change_policy": "起飞前 2h 免费改签一次", "refund": "起飞前 2h 收取 10% 退票费"}'),

            AssistantMessage(
                content="京都本周末晴空万里，最高24℃最低17℃，紫外线指数7，记得全天补涂SPF30+；500美元在机场换得76150日元零手续费，还附赠地铁一日券95折券；羽田07:10→伊丹08:05的JL2201仅12500日元，含20kg行李、免费Wi-Fi与茶点，返程再享9折。早起航班+防晒，旅途愉快！"),
        ]

        # 添加剩余未批量提交的消息
        if messages_batch:
            await context.add_messages(messages_batch)

        # 4. 获取上下文窗口：增大window_size，能看到更多压缩后的消息
        window = await context.get_context_window(window_size=50)

        # 5. 优化输出：打印消息类型+内容摘要，便于查看压缩效果
        print("=== 上下文窗口消息（压缩后）===")
        for idx, msg in enumerate(window.get_messages()):
            msg_type = type(msg).__name__
            content = msg.content
            #[:50] + "..." if len(msg.content) > 50 else msg.content
            print(f"[{idx + 1}] {msg_type}: {content}")

        print(f"\n=== 统计信息 ===")
        print(f"总消息数：{len(window.get_messages())}")
        print(f"Token统计：{window.statistic}")
    asyncio.run(run())