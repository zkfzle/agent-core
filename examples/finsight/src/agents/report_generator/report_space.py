import asyncio

from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.utils.llm.messages import HumanMessage

from src.agents.base import CodeSpace
from src.common.react_utils import ActionResult
from src.tools import SearchResult, ClickResult


class ReportSpace(CodeSpace):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.actions.update({
            "outline": self._handle_outline_action,
            "draft": self._handle_report_action,
            "report": self._handle_report_action,
            "search": self._handle_search_action
        })

        self.task = ""

    @staticmethod
    async def _handle_outline_action(action):
        return ActionResult(
            is_finished=True,
            message=HumanMessage(content=action.content)
        )

    @staticmethod
    async def _handle_report_action(action):
        return ActionResult(
            is_finished=True,
            message=HumanMessage(content=action.content)
        )

    async def _handle_search_action(self, action):
        response = await Runner.run_agent("deepsearch_agent", {"query": action.content, "task": self.task})
        self.cache.extend(response["cache"])
        return ActionResult(
            message=HumanMessage(content=response["final_result"])
        )

    # todo search？
    # async def _handle_search_action(self, action):

    async def initialize(self, inputs):
        self.task = inputs["task"]

        collect_data_list = [r for r in inputs.get("collected_data_list") if not (isinstance(r, SearchResult) or isinstance(r, ClickResult))]
        analysis_result_list = inputs.get("analysis_result_list")

        def _get_data(data_id: int):
            """Get dataset by index"""
            if 0 <= data_id < len(collect_data_list):
                return collect_data_list[data_id].data
            else:
                raise ValueError(f"Invalid data_id: {data_id}. Available range: 0-{len(collect_data_list) - 1}")

        def _get_analysis_result(data_id: int):
            """Get analysis results matching the query"""
            # Use LLM-based selection to find relevant analysis results
            if 0 <= data_id < len(analysis_result_list):
                return str(analysis_result_list[data_id])[:3000]
            else:
                raise ValueError(f"Invalid data_id: {data_id}. Available range: 0-{len(analysis_result_list) - 1}")

        def _get_deepsearch_result(query: str):
            """Call deep search agent"""
            response = asyncio.run(Runner.run_agent("deepsearch_agent", {"query": query, "task": inputs["task"]}))
            self.cache.extend(response["cache"])
            return response["final_result"]

        self.code_executor.set_variable("get_data", _get_data)
        self.code_executor.set_variable("get_analysis_result", _get_analysis_result)
        self.code_executor.set_variable("get_data_from_deep_search", _get_deepsearch_result)
