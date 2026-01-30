from openjiuwen.core.utils.llm.messages import HumanMessage

from src.agents.report_generator.report_space import ReportSpace
from src.common.agent_config import AgentRuntimeConfig, AgentConfig
from src.common.clients.openai_client import OpenAIClient
from src.common.react_utils import async_run_react_loop
from src.tools import SearchResult, ClickResult
from src.utils import extract_markdown
from src.utils.prompt_loader import get_prompt_loader


class SectionWriter:
    def __init__(self, config: AgentRuntimeConfig, llm_client):
        self._config = config
        self.llm_client = llm_client

        self.prompt_loader = get_prompt_loader('report_generator', report_type=self._config.get_agent_config().metadata.get('target_type'))
        self.max_iterations = self._config.get_agent_config().metadata.get('max_iterations', 5)
        self.target_language = self._config.get_agent_config().metadata.get('target_language')
        self.enable_chart = self._config.get_agent_config().metadata.get('enable_chart', True)

        self.SECTION_PROMPT_TEMPLATE = self.prompt_loader.get_prompt("section_writing") if self.enable_chart else self.prompt_loader.get_prompt("section_writing_wo_chart")
        self.DATA_API = self.prompt_loader.get_prompt("data_api")
        self.FINAL_POLISH_PROMPT = self.prompt_loader.get_prompt("final_polish")

    def config(self):
        return self._config

    async def invoke(self, inputs, runtime):
        action_space = ReportSpace(
            working_dir=self._config.get_agent_config().metadata.get('working_dir'),
            config=self._config.get_agent_config().metadata.get('config'),
        )
        await action_space.initialize(inputs)

        collect_data_list = inputs["collected_data_list"]
        collect_data_list = [d for d in collect_data_list if
                             not (isinstance(d, SearchResult) or isinstance(d, ClickResult))]
        analysis_result_list = inputs["analysis_result_list"]
        data_info = "\n\n## Available Datas\n\n"
        for idx, item in enumerate(collect_data_list):
            data_info += f"**Data ID {idx}:**\n{item.brief_str()}\n\n"
        data_info += "\nYou can access these datasets using `get_data(data_id)` in your code.\n"
        data_info += "\n\n## Available Analysis Reports\n\n"
        for idx, item in enumerate(analysis_result_list):
            data_info += f"**Analysis Report ID {idx}:**\n{item.brief_str()}\n\n"
        data_info += "\nYou can access these analysis reports using `get_analysis_result(analysis_result_id)` in your code.\n"
        user_prompt = self.SECTION_PROMPT_TEMPLATE.format(
            task=inputs['task'],
            report_theme=inputs['task'],
            section_description=inputs['section_outline'],
            data_api=self.DATA_API,
            data_info=data_info,
            max_iterations=self.max_iterations,
            target_language=self.target_language,
        )
        messages = [HumanMessage(content=user_prompt)]
        messages, reach_max_iter = await async_run_react_loop(self.llm_client, messages, max_iterations=self.max_iterations, action_space=action_space)

        draft_section = messages[-1].content
        all_analysis_result = inputs["analysis_result_list"]
        all_image_list = []
        for analysis_result in all_analysis_result:
            all_image_list.extend(analysis_result.get_all_img())
        reference_image = '\n'.join(all_image_list)
        final_prompt = self.FINAL_POLISH_PROMPT.format(
            draft_report=draft_section,
            reference_image=reference_image,
            target_language=self.target_language
        )
        final_message = [HumanMessage(content=final_prompt)]
        output = await self.llm_client.ainvoke(messages=final_message)
        final_section = extract_markdown(output.content)
        return {
            "final_section": final_section,
            "cache": action_space.cache,
        }

    @classmethod
    def get_provider(cls, config):
        def provider():
            agent_config = AgentConfig(
                id="section_writer",
                name="section_writer",
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
