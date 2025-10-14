"""ReActAgent"""
from typing import Dict, Any, List, AsyncIterator

from jiuwen.agent.config.base import LLMCallConfig
from jiuwen.agent.config.chat_config import ChatAgentConfig
from jiuwen.core.agent.agent import Agent
from jiuwen.core.runtime.config import Config
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.utils.tool.base import Tool
from jiuwen.core.utils.llm_call.base import LLMCall
from jiuwen.core.utils.llm.hash_util import generate_key
from jiuwen.core.utils.llm.model_utils.model_factory import ModelFactory
from jiuwen.core.context_engine.config import ContextEngineConfig



def create_chat_agent_config(agent_id: str,
                             agent_version: str,
                             description: str,
                             model: LLMCallConfig
                             ):
    config = ChatAgentConfig(id=agent_id,
                             version=agent_version,
                             description=description,
                             model=model
                             )
    return config


def create_chat_agent(agent_config: ChatAgentConfig,
                      tools: List[Tool] = None):
    agent = ChatAgent(agent_config)
    agent.bind_tools(tools or [])
    return agent


class ChatAgent(Agent):
    def __init__(self, agent_config: ChatAgentConfig):
        self._config = Config()
        self._config.set_agent_config(agent_config=agent_config)
        super().__init__(self._config)
        self.context_engine = self._create_context_engine()
        llm_config = agent_config.model
        self._llm_call = LLMCall(
            llm_config.model.model_info.model_name,
            self._init_model(llm_config.model),
            llm_config.system_prompt,
            llm_config.user_prompt,
            llm_config.freeze_system_prompt,
            llm_config.freeze_user_prompt
        )

    def _init_model(self, model_config):
        model_id = generate_key(
            model_config.model_info.api_key,
            model_config.model_info.api_base,
            model_config.model_provider
        )

        model = self._runtime.get_model(model_id=model_id)

        if model is None:
            model = ModelFactory().get_model(
                model_provider=model_config.model_provider,
                api_base=model_config.model_info.api_base,
                api_key=model_config.model_info.api_key
            )
            self._runtime.add_model(model_id=model_id, model=model)

        return self._runtime.get_model(model_id=model_id)

    def _create_context_engine(self) -> ContextEngine:
        """创建ContextEngine实例"""
        context_config = ContextEngineConfig()
        return ContextEngine(
            agent_id=self._config.get_agent_config().id,
            config=context_config,
        )

    async def invoke(self, inputs: Dict) -> Dict:
        # 1. init ContextEngine and Runtime
        session_id = inputs.pop("conversation_id", "default_session")
        runtime = await self._runtime.pre_run(session_id=session_id)

        # 2. invoke LLMCall
        agent_context = self.context_engine.get_agent_context(session_id)
        result = await self._llm_call.invoke(
            inputs=inputs,
            runtime=runtime,
            history=agent_context.get_messages(),
            tools=self._runtime.get_tool_info()
        )
        await runtime.post_run()
        return dict(output=result.content, tool_calls=result.tool_calls)

    async def stream(self, inputs: Dict) -> AsyncIterator[Any]:
        # 1. init ContextEngine and Runtime
        session_id = inputs.pop("conversation_id", "default_session")
        runtime = await self._runtime.pre_run(session_id=session_id)

        # 2. stream invoke LLMCall
        agent_context = self.context_engine.get_agent_context(session_id)
        stream_iterator = self._llm_call.stream(
            inputs=inputs,
            runtime=runtime,
            history=agent_context.get_messages(),
            tools=self._runtime.get_tool_info()
        )
        await runtime.post_run()
        async for result in stream_iterator:
            yield dict(output=result.content, tool_calls=result.tool_calls)

    def get_llm_calls(self) -> Dict:
        return dict(llm_call=self._llm_call)

    def copy(self) -> "Agent":
        return create_chat_agent(self._config.get_agent_config())