import asyncio
import os
import pickle
import sys
import traceback

import yaml

from src.agents.data_analyzer.chart_drawer import ChartDrawer
from src.agents.data_analyzer.report_analyzer import ReportAnalyzer
from src.agents.data_collector.data_collector import DataCollector
from src.agents.report_generator.outline_generator import OutlineGenerator
from src.agents.report_generator.section_writer import SectionWriter
from src.agents.search_agent.search_agent import DeepSearchAgent
from src.common.clients.async_embedding_client import EmbeddingClient
from src.common.clients.openai_client import OpenAIClient
from src.common.controller import Controller

from openjiuwen.core.utils.tool.tool import LocalFunction
from openjiuwen.core.utils.tool.param import Param
from openjiuwen.core.runner.runner import resource_mgr, Runner
from openjiuwen.core.common.logging import logger

from src.common.runner import get_and_register_local_tools
from src.tools import get_tool_categories, get_tool_by_name
from src.agents.finsight_agent.finsight_agent import FinSightAgent, FinSightWorkspace

from src.config import Config
from dotenv import load_dotenv
load_dotenv()

os.environ["LLM_SSL_VERIFY"] = "False"


def load_config_with_env(config_path):
    """加载配置文件并替换环境变量"""
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换所有 ${VAR_NAME} 格式的环境变量
    for key, value in os.environ.items():
        placeholder = f"${{{key}}}"
        content = content.replace(placeholder, value)

    return content


async def run_report(resume=True):
    try:
        config = Config(
            config_file_path='my_config.yaml',
        )

        tools = get_and_register_local_tools()
        Runner.add_agent("data_collector", DataCollector.get_provider(config))
        Runner.add_agent("deepsearch_agent", DeepSearchAgent.get_provider(config))
        Runner.add_agent("report_analyzer", ReportAnalyzer.get_provider(config))
        Runner.add_agent("chart_drawer", ChartDrawer.get_provider(config))
        Runner.add_agent("outline_generator", OutlineGenerator.get_provider(config))
        Runner.add_agent("section_writer", SectionWriter.get_provider(config))

        llm_client = OpenAIClient(**config.config.get("llm_config_list")[0])
        embedding_client = EmbeddingClient(**config.config.get("llm_config_list")[1])
        vlm_client = OpenAIClient(**config.config.get("llm_config_list")[2])

        workspace = FinSightWorkspace(
            target_type=config.config.get("target_type"),
            custom_collective_tasks=config.config.get("custom_collect_tasks"),
            custom_analysis_tasks=config.config.get("custom_analysis_tasks"),
            target_name=config.config.get("target_name"),
            stock_code=config.config.get("stock_code"),
            enable_chart=True,
            add_reference=True,
            reference_doc_path=config.config.get('reference_doc_path'),
            working_dir=config.config.get("output_dir"),
            image_save_dir=config.config.get("output_dir") + "/images",
            outline_template_path=config.config.get("outline_template_path")
        )

        if not os.path.exists(config.config.get("output_dir")):
            os.makedirs(config.config.get("output_dir"))
        if not os.path.exists(config.config.get("output_dir") + "/images"):
            os.mkdir(config.config.get("output_dir") + "/images")

        task_controller = Controller()

        config.config["name"] = "FinSightAgent"
        main_agent = FinSightAgent(config.config, llm_client, controller=task_controller, embedding_client=embedding_client, vlm_client=vlm_client, tools=tools)

        await main_agent.invoke(workspace.model_dump())

    except Exception as e:
        logger.error(f"Exception in run_report: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise e


if __name__ == '__main__':
    try:
        asyncio.run(run_report(resume=True))
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        sys.exit(130)
    except Exception as e:
        raise e