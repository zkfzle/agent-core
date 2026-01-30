import asyncio
import os.path
import pickle
import uuid
from typing import Any, List

from openjiuwen.core.runner.runner import Runner
from pydantic import BaseModel

from src.agents.data_analyzer.data_analyzer import DataAnalyzer
from src.agents.report_generator.report_generator import ReportGenerator
from src.common.event import PriorityEvent
from openjiuwen.core.common.logging import logger

from src.agents.finsight_agent.analysis_event import AnalysisEvent
from src.agents.finsight_agent.collective_event import CollectiveEvent


class FinSightWorkspace(BaseModel):
    target_type: str
    target_name: str
    stock_code: str
    enable_chart: bool
    add_reference: bool
    reference_doc_path: str
    working_dir: str
    image_save_dir: str
    outline_template_path: str
    custom_collective_tasks: List[str]
    custom_analysis_tasks: List[str]

    analysis_task: str = None
    all_collective_tasks: List[PriorityEvent] = None
    all_analysis_tasks: List[PriorityEvent] = None


class FinSightAgent:
    def __init__(self, config, llm_client, controller, embedding_client, vlm_client, tools):
        self.config = config
        self.llm_client = llm_client
        self.vlm_client = vlm_client
        self.controller = controller
        self.embedding_client = embedding_client
        self.tools = tools

        self.invoke_map = {
            "DataCollector": self.create_collection_task_executor,
            "DataAnalyzer": self.create_analysis_task_executor,
            "ReportGenerator": self.create_report_task_executor,
        }

        self._state = {}

    def create_collection_task_executor(self):
        async def run_data_collector_task(task):
            result = await Runner.run_agent("data_collector", task.inputs)
            task.result = result
            task.status = "finished"
            self.controller.refresh()
            return result
        return run_data_collector_task

    def create_analysis_task_executor(self):
        async def run_data_analyzer_task(task):
            agent = DataAnalyzer(self.config)
            await agent.invoke(task)
            if task.result and task.status == "finished":
                self.controller.refresh()
            return task.result
        return run_data_analyzer_task

    def create_report_task_executor(self):
        async def run_report_generator_task(task):
            agent = ReportGenerator(self.config)
            await agent.invoke(task)
            task.status = "finished"
            self.controller.refresh()
            return task.result

        return run_report_generator_task

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self._state, f)

    def load(self, path):
        try:
            with open(path, "rb") as f:
                self._state = pickle.load(f)
        except Exception as e:
            logger.error("No checkpoint founded!")

    async def invoke(self, inputs):
        task_type = inputs.get('target_type', 'company')
        collect_tasks = inputs.get('custom_collective_tasks')
        analysis_tasks = inputs.get('custom_analysis_tasks')
        target_name = inputs.get("target_name")
        stock_code = inputs.get("stock_code")

        checkpoint_path = os.path.join(inputs["working_dir"], "finsight.pkl")
        self.load(checkpoint_path)
        controller_state = self._state.get("controller_state")

        if controller_state:
            self.controller.load_from_state(controller_state)
        else:
            research_query = f"Research target: {target_name} (ticker: {stock_code}), target type: {task_type}"
            collect_tasks = [CollectiveEvent.from_description(t, "DataCollector", task_type=task_type) for t in collect_tasks]
            analysis_tasks = [AnalysisEvent.from_description(t, "DataAnalyzer", task_type=task_type) for t in analysis_tasks]
            collective_task_coroutines = CollectiveEvent.async_create_tasks(
                goal=research_query,
                assignee="DataCollector",
                llm=self.llm_client,
                task_type=task_type,
                max_num=5,
                existing_tasks=collect_tasks,
            )
            analysis_task_coroutines = AnalysisEvent.async_create_tasks(
                goal=research_query,
                assignee="DataAnalyzer",
                llm=self.llm_client,
                task_type=task_type,
                max_num=5,
                existing_tasks=analysis_tasks,
            )

            generated_collect_tasks, generated_analysis_tasks = await asyncio.gather(
                collective_task_coroutines,
                analysis_task_coroutines
            )

            self.controller.add_tasks(collect_tasks)
            self.controller.add_tasks(
                generated_collect_tasks,
                lambda_filter=lambda task_1, task_2: task_1.task_description != task_2.task_description
            )

            self.controller.add_tasks(analysis_tasks)
            self.controller.add_tasks(
                generated_analysis_tasks,
                lambda_filter=lambda task_1, task_2: task_1.task_description != task_2.task_description
            )

            self.controller.add_tasks([
                PriorityEvent(
                    id=uuid.uuid4().hex[:8],
                    assignee="ReportGenerator",
                    priority=3,
                    type=task_type,
                    task_description=f'Research target: {target_name} (ticker: {stock_code})',
                    status="pending"
                )
            ])
            self._state.update({
                "controller_state": self.controller.state(),
            })


        tasks_to_run = self.controller.get_next_parallel_tasks()
        while tasks_to_run:
            collected_data_list = self._state.get("collected_data_list")
            if not collected_data_list and tasks_to_run[0].assignee in ["ReportGenerator", "DataAnalyzer"]:
                collected_data_list = []
                for t in self.controller.past_tasks:
                    if t.assignee == "DataCollector":
                        collected_data_list.extend(t.result["cache"])
                self._state.update({
                    "collected_data_list": collected_data_list,
                })
                self.save(checkpoint_path)

            analysis_data_list = self._state.get("analysis_result_list")
            if not analysis_data_list and tasks_to_run[0].assignee == "ReportGenerator":
                analysis_data_list = []
                for t in self.controller.past_tasks:
                    if t.assignee == "DataAnalyzer":
                        analysis_data_list.append(t.result)
                self._state.update({
                    "analysis_result_list": analysis_data_list,
                })
                self.save(checkpoint_path)

            logger.info(f"[FinSight] Running {len(tasks_to_run)} {tasks_to_run[0].assignee}s in total.")
            for task_to_run in tasks_to_run:
                task_to_run.callable = self.invoke_map[task_to_run.assignee]()
                task_to_run.inputs = {
                    **inputs,
                    "task": f"Research target: {target_name} (ticker: {stock_code}), task: {task_to_run.task_description}" if task_to_run.assignee == "DataCollector" else f'Research target: {inputs["target_name"]} (ticker: {inputs["stock_code"]})',
                    'analysis_task': task_to_run.task_description,
                    "target_type": inputs.get('target_type', 'company'),
                    'collected_data_list': collected_data_list,
                    'analysis_result_list': analysis_data_list,
                }

            await self.controller.run_parallel_tasks()
            self._state.update({
                "controller_state": self.controller.state(),
            })
            self.save(checkpoint_path)
            tasks_to_run = self.controller.get_next_parallel_tasks()

        return