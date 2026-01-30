import asyncio
from datetime import datetime
from typing import List, Any

from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.utils.llm.messages import HumanMessage

from src.common.agent_config import AgentRuntimeConfig, AgentConfig
from src.common.clients.openai_client import OpenAIClient
from src.common.react_utils import async_run_react_loop
from src.tools import ToolResult, get_tool_categories
from src.utils.prompt_loader import get_prompt_loader
from src.agents.base import CodeSpace


class CollectorSpace(CodeSpace):
    def __init__(self, working_dir: str, config, tools: List[str] = None, agents: List[str] = None):
        super().__init__(working_dir, tools, agents)
        self.code_executor.set_variable("save_result", self._save_result)
        self.code_executor.set_variable("call_tool", self._execute_with_code)

        self.config = config

        for tool_type, tool_name_list in get_tool_categories().items():
            if tool_type != "web":
                self.tools.extend(tool_name_list)
        self.agents = ["deepsearch_agent"]

    def _execute_with_code(self, tool_name: str = None, **kwargs):
        if tool_name in self.tools:
            response = asyncio.run(Runner.run_tool(tool_name, kwargs))
            sources = [item.source for item in response]
            data_list = [item.data for item in response]
            sources = "\n".join(sources)
            import sys
            display_note = f"[Tool Result Overview] Gather {len(response)} Tool Results.\n"
            for i, item in enumerate(response):
                display_note += f"-{i}. Name: {item.name}\nSource: {item.source}\n"
            print(f"\n\n{display_note}\n\n", file=sys.stdout, flush=True)
            return data_list

        elif tool_name in self.agents:
            response = asyncio.run(Runner.run_agent("deepsearch_agent", kwargs))
            self.cache.extend(response["cache"])
            return response["final_result"]
        else:
            return []

    def _save_result(self, var: Any, result_name: str, result_description: str, data_source: str):
        self.cache.append(
            ToolResult(
                name=result_name,
                description=result_description,
                data=var,
                source=data_source
            )
        )


class DataCollector:
    def __init__(self, config: AgentRuntimeConfig, llm_client):
        self._config = config
        self.llm_client = llm_client

        self.prompt_loader = get_prompt_loader('data_collector', report_type='general')
        self.DATA_COLLECT_PROMPT = self.prompt_loader.get_prompt('data_collect')

        self.max_iterations = self._config.get_agent_config().metadata.get('max_iterations', 5)

    def config(self):
        return self._config

    async def invoke(self, inputs, runtime):
        action_space = CollectorSpace(
            working_dir=self._config.get_agent_config().metadata.get('working_dir'),
            config=self._config.get_agent_config().metadata.get('config'),
        )

        task = inputs.get('task')
        if not task:
            raise ValueError("Input data must contain a 'task' key.")

        # Extract research target from task
        target_name = inputs.get('target_name', '')
        stock_code = inputs.get('stock_code', '')
        research_target = f"{target_name} (ticker: {stock_code})"

        prompt_args = {
            "api_descriptions": action_space.get_api_descriptions(),
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task": task,
            "target_language": self._config.get_agent_config().metadata.get('target_language'),
            "research_target": research_target
        }

        user_prompt = self.DATA_COLLECT_PROMPT.format(**prompt_args)
        messages = [HumanMessage(content=user_prompt)]
        messages, reach_max_iter = await async_run_react_loop(self.llm_client, messages, max_iterations=self.max_iterations, action_space=action_space)
        final_result = messages[-1].content
        return {
            "final_result": final_result,
            "cache": action_space.cache,
        }

    @classmethod
    def get_provider(cls, config):
        def provider():
            agent_config = AgentConfig(
                id="data_collector",
                name="data_collector",
                description="a agent that can collect data from the internet and variable apis",
                metadata={
                    "target_language": config.config["language"],
                    "max_iterations": 20,
                    "working_dir": config.config["working_dir"],
                    "config": config
                }
            )

            llm_client = OpenAIClient(**config.config["llm_config_list"][0])
            return cls(config=AgentRuntimeConfig(agent_config), llm_client=llm_client)
        return provider
