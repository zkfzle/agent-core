from typing import List
from pydantic import BaseModel, Field
from openjiuwen.core.runtime.config import Config


class AgentConfig(BaseModel):
    id: str
    name: str
    description: str
    metadata: dict = Field(default_factory=dict)


class AgentRuntimeConfig(Config):
    def __init__(self, config) -> None:
        super().__init__()
        self.set_agent_config(config)