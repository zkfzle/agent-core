from typing import List, Dict, Optional

from pydantic import Field

from jiuwen.agent.config.base import AgentConfig, LLMCallConfig


class ChatAgentConfig(AgentConfig):
    model: LLMCallConfig = Field(default=LLMCallConfig())
