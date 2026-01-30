import re
from datetime import datetime

import json_repair
from openjiuwen.core.utils.llm.messages import HumanMessage

from src.agents.search_agent.search_agent import DeepSearchResult
from src.common.agent_config import AgentRuntimeConfig, AgentConfig
from src.common.clients.openai_client import OpenAIClient
from src.common.react_utils import async_run_react_loop
from src.tools import ToolResult, get_tool_categories, SearchResult, ClickResult
from src.utils.prompt_loader import get_prompt_loader
from src.agents.data_analyzer.analyzer_space import AnalyzerSpace



class ReportAnalyzer:
    def __init__(self, config: AgentRuntimeConfig, llm_client):
        self._config = config
        self.llm_client = llm_client

        self.max_iterations = self._config.get_agent_config().metadata.get('max_iterations', 5)
        self.enable_chart = self._config.get_agent_config().metadata.get('enable_chart', True)
        self.target_language = self._config.get_agent_config().metadata.get('target_language')

        self.prompt_loader = None

    def config(self):
        return self._config

    def _prepare_prompt(self, inputs):
        target_type = inputs["target_type"]
        analysis_task = inputs["analysis_task"]
        self.prompt_loader = get_prompt_loader('data_analyzer', report_type=target_type)

        collect_data_list = inputs.get("collected_data_list")
        collect_data_list = [d for d in collect_data_list if not isinstance(d, SearchResult) and not isinstance(d, ClickResult)]
        collect_data_list = [d for d in collect_data_list if not isinstance(d, DeepSearchResult)]

        if self.enable_chart:
            prompt = self.prompt_loader.get_prompt("data_analysis")
        else:
            prompt = self.prompt_loader.get_prompt("data_analysis_wo_chart")

        data_api = self.prompt_loader.get_prompt("data_api")
        data_info = self._format_collect_data(analysis_task, collect_data_list)
        return prompt.format(
            api_descriptions=data_api,
            data_info=data_info,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_query=analysis_task,
            target_language=self.target_language,
        )

    @staticmethod
    def _format_collect_data(analysis_task, collect_data_list):
        formatted_data = ""
        for idx, item in enumerate(collect_data_list):
            formatted_data += f"Data (id:{idx}):\n{item.brief_str()}\n\n"
        return formatted_data


    async def invoke(self, inputs, runtime):
        action_space = AnalyzerSpace(
            working_dir=self._config.get_agent_config().metadata.get('working_dir'),
            config=self._config.get_agent_config().metadata.get('config'),
        )
        await action_space.initialize(inputs)

        user_prompt = self._prepare_prompt(inputs)

        messages = [HumanMessage(content=user_prompt)]
        messages, reach_max_iter = await async_run_react_loop(self.llm_client, messages, max_iterations=self.max_iterations, action_space=action_space)

        if reach_max_iter:
            conversation_history = [item.content for item in messages]
            analysis_info = "\n\n".join(conversation_history)
            prompt = self.prompt_loader.get_prompt("report_draft") if self.enable_chart else self.prompt_loader.get_prompt("report_draft_wo_chart")

            prompt = prompt.format(
                current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                analysis_info=analysis_info,
                target_language=self.target_language,
            )
            response = await self.llm_client.ainvoke(
                messages=[HumanMessage(content=prompt)],
                response_format={"type": "json_object"}
            )
            match = re.search(r'```json([\s\S]*?)```', response.content)
            if match:
                response = match.group(1).strip()
            try:
                report = json_repair.loads(response.content)
                report_title = report["title"]
                report_content = report["content"]
                final_result = f'# {report_title}\n{report_content}'
            except Exception:
                final_result = response

        else:
            final_result = messages[-1].content
        return {
            "final_result": final_result,
        }

    @classmethod
    def get_provider(cls, config):
        def provider():
            agent_config = AgentConfig(
                id="report_analyzer",
                name="report_analyzer",
                description="a agent that can collect data from the internet and variable apis",
                metadata={
                    "target_language": config.config["language"],
                    "max_iterations": 3,
                    "working_dir": config.config["working_dir"],
                    "config": config
                }
            )

            llm_client = OpenAIClient(**config.config["llm_config_list"][0])
            return cls(config=AgentRuntimeConfig(agent_config), llm_client=llm_client)
        return provider
