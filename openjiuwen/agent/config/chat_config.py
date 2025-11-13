from pydantic import Field

from openjiuwen.agent.config.base import AgentConfig, LLMCallConfig


class ChatAgentConfig(AgentConfig):
    model: LLMCallConfig = Field(default=LLMCallConfig())
