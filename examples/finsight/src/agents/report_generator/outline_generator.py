import asyncio
import os
import uuid
from datetime import datetime

from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.utils.llm.messages import HumanMessage

from src.agents.base import CodeSpace
from src.agents.report_generator.report_class import Report
from src.agents.report_generator.report_space import ReportSpace
from src.agents.search_agent.search_agent import DeepSearchAgent
from src.common.agent_config import AgentRuntimeConfig, AgentConfig
from src.common.clients.openai_client import OpenAIClient
from src.common.react_utils import ActionResult, async_run_react_loop
from src.tools import SearchResult, ClickResult
from src.utils import extract_markdown
from src.utils.prompt_loader import get_prompt_loader



class OutlineGenerator:
    def __init__(self, config: AgentRuntimeConfig, llm_client):
        self._config = config
        self.llm_client = llm_client

        self.prompt_loader = get_prompt_loader('report_generator', report_type=self._config.get_agent_config().metadata.get('target_type'))
        self.OUTLINE_DRAFT_PROMPT = self.prompt_loader.get_prompt('outline_draft')
        self.DATA_API_PROMPT = self.prompt_loader.get_prompt('data_api_outline')

        outline_template_path = self._config.get_agent_config().metadata.get('outline_template_path')
        outline_template = ""
        if outline_template_path is None or not os.path.exists(outline_template_path):
            outline_template = ""
        else:
            with open(outline_template_path, 'r', encoding='utf-8') as f:
                outline_template = f.read()
        self.OUTLINE_TEMPLATE = outline_template

        self.max_iterations = self._config.get_agent_config().metadata.get('max_iterations', 5)
        self.target_language = self._config.get_agent_config().metadata.get('target_language')

    def config(self):
        return self._config

    async def invoke(self, inputs, runtime):
        action_space = ReportSpace(
            working_dir=self._config.get_agent_config().metadata.get('working_dir'),
            config=self._config.get_agent_config().metadata.get('config'),
        )
        await action_space.initialize(inputs)
        analysis_result_list = inputs.get("analysis_result_list")
        data_info = "You have access to the following analysis results:\n\n"
        for idx, result in enumerate(analysis_result_list):
            data_info += f"**Analysis Report ID {idx}:**\n{result.brief_str()}\n\n"
        data_info += "\nYou can retrieve detailed content using `get_analysis_result(analysis_id)` in your code.\n"

        user_prompt = self.OUTLINE_DRAFT_PROMPT.format(
            task=inputs['task'],
            report_requirements=self.OUTLINE_TEMPLATE,
            data_api=self.DATA_API_PROMPT,
            data_info=data_info,
            max_iterations=self.max_iterations,
            target_language=self.target_language,
        )
        messages = [HumanMessage(content=user_prompt)]
        messages, reach_max_iter = await async_run_react_loop(self.llm_client, messages, max_iterations=self.max_iterations, action_space=action_space)
        final_result = messages[-1].content

        outline_content = extract_markdown(final_result)
        report = Report(outline_content) if outline_content else Report("# Error: Could not generate outline")
        return {
            "final_result": final_result,
            "cache": action_space.cache,
            "report": report,
        }

    @classmethod
    def get_provider(cls, config):
        def provider():
            agent_config = AgentConfig(
                id="outline_generator",
                name="outline_generator",
                description="a agent that can collect data from the internet and variable apis",
                metadata={
                    "target_language": config.config["language"],
                    "max_iterations": 10,
                    "working_dir": config.config["working_dir"],
                    "config": config,
                    "outline_template_path": config.config["outline_template_path"],
                    "target_type": config.config["target_type"],
                }
            )

            llm_client = OpenAIClient(**config.config["llm_config_list"][0])
            return cls(config=AgentRuntimeConfig(agent_config), llm_client=llm_client)
        return provider
