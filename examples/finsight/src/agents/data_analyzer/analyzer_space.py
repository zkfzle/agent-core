import asyncio
import uuid
from typing import List

from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.utils.llm.messages import HumanMessage

from src.agents.base import CodeSpace
from src.agents.search_agent.search_agent import DeepSearchAgent
from src.common.react_utils import ActionResult


class AnalyzerSpace(CodeSpace):
    def __init__(self, working_dir: str, config, tools: List[str] = None, agents: List[str] = None):
        super().__init__(working_dir, tools, agents)
        self.config = config

        self.actions.update({
            "report": self._handle_report_action
        })

    @staticmethod
    async def _handle_report_action(action):
        return ActionResult(
            is_finished=True,
            message=HumanMessage(content=action.content),
        )

    async def initialize(self, inputs):
        collect_data_list = inputs["collected_data_list"]
        def _get_existed_data(data_id: int):
            return collect_data_list[data_id].data

        def _get_deepsearch_result(query: str):
            response = asyncio.run(Runner.run_agent("deepsearch_agent", {"query": query, "task": inputs["task"]}))
            self.cache.extend(response["cache"])
            return response["final_result"]

        self.code_executor.set_variable("session_output_dir", inputs.get("image_save_dir"))
        self.code_executor.set_variable("collect_data_list", [item.data for item in collect_data_list])
        self.code_executor.set_variable("get_data_from_deep_search", _get_deepsearch_result)
        self.code_executor.set_variable("get_existed_data", _get_existed_data)

        custom_palette = [
            "#8B0000",  # deep crimson
            "#FF2A2A",  # bright red
            "#FF6A4D",  # orange-red
            "#FFDAB9",  # pale peach
            "#FFF5E6",  # cream
            "#FFE4B5",  # beige
            "#A0522D",  # sienna
            "#5C2E1F",  # dark brown
        ]
        self.code_executor.set_variable("custom_palette", custom_palette)
        await self.code_executor.execute("import seaborn as sns\nsns.set_palette(custom_palette)")



