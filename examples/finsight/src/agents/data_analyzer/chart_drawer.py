import os
import re

from openjiuwen.core.utils.llm.messages import HumanMessage, AIMessage, BaseMessage
from threading import Semaphore
from src.agents.data_analyzer.analyzer_space import AnalyzerSpace
from src.common.agent_config import AgentConfig, AgentRuntimeConfig
from src.common.clients.openai_client import OpenAIClient
from src.common.react_utils import Action
from src.utils.helper import image_to_base64
from src.utils.logger import get_logger
from src.utils.prompt_loader import get_prompt_loader

logger = get_logger()


class ChartDrawer:
    def __init__(self, config, llm_client, vlm_client):
        self._config = config
        self.llm_client = llm_client
        self.vlm_client = vlm_client

        self.DRAW_CHART_PROMPT = None
        self.VLM_CRITIQUE_PROMPT = None
        self.code_executor = None
        self.llm_parser = None

        self.working_dir = self._config.get_agent_config().metadata.get('working_dir')
        self.image_save_dir = os.path.join(self.working_dir, "images")
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)

    def config(self):
        return self._config

    async def invoke(self, inputs, runtime):
        target_type = inputs["target_type"]

        prompt_loader = get_prompt_loader('data_analyzer', report_type=target_type)
        self.DRAW_CHART_PROMPT = prompt_loader.get_prompt("draw_chart")
        self.VLM_CRITIQUE_PROMPT = prompt_loader.get_prompt("vlm_critique")

        action_space = AnalyzerSpace(
            working_dir=self._config.get_agent_config().metadata.get('working_dir'),
            config=self._config.get_agent_config().metadata.get('config'),
        )
        if inputs.get("code_executor"):
            action_space.code_executor = inputs["code_executor"]
        inputs = {
            **inputs,
            "image_save_dir": self.image_save_dir,
        }
        await action_space.initialize(inputs)
        self.code_executor = action_space.code_executor
        self.llm_parser = action_space.parse_llm_response
        return await self._draw_chart(inputs, runtime)


    async def _draw_chart(self, input_data, runtime, max_iterations: int = 3):
        report_content = input_data["report_content"]
        analysis_task = input_data['analysis_task']
        chart_names = re.findall(r'@import\s+"(.*?)"', report_content)
        image_save_dir = self.image_save_dir
        current_variables = self.code_executor.get_environment_info()

        name_mapping = {}  # long chart name -> short filename
        name_description_mapping = {}  # long chart name -> description
        chart_code_mapping = {}  # long chart name -> code snippet

        # Concurrency control semaphore
        charts_completed = set()
        for long_chart_name in chart_names:
            if long_chart_name in charts_completed:
                continue
            # TODO: Shared environments need isolation; temporarily limit concurrency to 1
            with Semaphore(1):
                new_chart_code, new_chart_name = await self._draw_single_chart(
                    task=analysis_task,
                    report_content=report_content,
                    chart_name=long_chart_name,
                    image_save_dir=image_save_dir,
                    current_variables=current_variables,
                    max_iterations=max_iterations,
                    runtime=runtime
                )
                name_mapping[long_chart_name] = new_chart_name
                chart_code_mapping[long_chart_name] = new_chart_code
                charts_completed.add(long_chart_name)

        for long_chart_name, new_chart_name in name_mapping.items():
            chart_des = await self._generate_description(new_chart_name, input_data["image_save_dir"])
            name_description_mapping[long_chart_name] = chart_des

        return chart_code_mapping, name_mapping, name_description_mapping

    async def _generate_description(self, chart_name: str, image_save_dir) -> str:
        chart_name_path = os.path.join(image_save_dir, chart_name)
        image_b64 = image_to_base64(chart_name_path)
        if not image_b64:
            return ""

        messages = [
            {"role": "user", "content": [
                {"type": "text",
                 "text": "Give a short description  as the caption of this chart, explaining the key data points and takeaways. Your response should be less than 100 words. Don't output any other words."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]}
        ]
        response = await self.vlm_client.ainvoke(
            messages=messages
        )
        return response.content

    async def _draw_single_chart(
            self,
            task: str,
            report_content: str,
            chart_name: str,
            image_save_dir: str,
            current_variables: str,
            max_iterations: int = 3,
            runtime=None
    ) -> str:
        """
        Run iterative “code generation → VLM critique” cycles for a single chart.
        """

        init_prompt = self.DRAW_CHART_PROMPT.format(
            task=task,
            content=report_content,
            chart_name=chart_name,
            data=current_variables
        )

        conversation_history = [
            HumanMessage(content=init_prompt)
        ]

        last_successful_code = ""
        last_successful_chart_path = ""
        logger.info(f"Start drawing chart: {chart_name}")

        # --- Main VLM evaluation loop ---
        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}")

            # --- Phase 1: generate/execute code (up to 3 retries) ---
            chart_code, chart_filepath = await self._generate_and_execute_code(
                conversation_history, image_save_dir
            )
            logger.info(f"chart_code: {chart_code}")
            logger.info(f"chart_filepath: {chart_filepath}")
            if not chart_filepath:
                return last_successful_code, os.path.basename(
                    last_successful_chart_path) if last_successful_chart_path else ""
            logger.info("Image generation succeeded")
            last_successful_code = chart_code
            last_successful_chart_path = chart_filepath

            # --- Phase 2: VLM evaluation ---
            image_b64 = image_to_base64(chart_filepath)
            if not image_b64:
                return last_successful_code, os.path.basename(last_successful_chart_path)

            critic_response = await self.vlm_client.ainvoke(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.VLM_CRITIQUE_PROMPT.format(
                                task=task,
                                content=report_content,
                            )},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                        ]
                    }
                ],
            )
            critic_response = critic_response.content
            logger.info("Image critic succeeded")
            if 'finish' in critic_response.lower():
                if isinstance(last_successful_code, AIMessage):
                    last_successful_code = last_successful_code.content
                return last_successful_code, os.path.basename(last_successful_chart_path)
            if isinstance(last_successful_code, str):
                last_successful_code = AIMessage(content=last_successful_code)
            conversation_history.append(AIMessage(content=last_successful_code.content))
            feedback_for_llm = (
                "The chart above was produced from your previous code. A visualization expert shared the critique below:\n\n"
                f"{critic_response}\n\n"
                "Please write new Python code to address every issue and generate an improved chart. "
                f"Overwrite the previous file '{os.path.basename(last_successful_chart_path)}'."
            )
            conversation_history.append(HumanMessage(content=feedback_for_llm))
        if isinstance(last_successful_code, AIMessage):
            last_successful_code = last_successful_code.content
        return last_successful_code, os.path.basename(last_successful_chart_path)

    async def _generate_and_execute_code(self, conversation_history: list, image_save_dir) -> tuple[
        str | None, str | None]:
        """
        Attempt (up to three times) to generate and execute the chart code.

        Returns:
            (llm_response, chart_filepath) on success; otherwise (None, None).
        """
        for _ in range(3):  # internal retries
            logger.info(f"Generating code, attempt {_ + 1}")

            llm_response = await self.llm_client.ainvoke(
                messages=conversation_history,
                # stop=['</execute']
            )
            action = self.llm_parser(llm_response)
            if not action:
                conversation_history.append(AIMessage(content=llm_response.content))
                conversation_history.append(HumanMessage(
                    content="Your reply did not include a valid <execute> code block. Please provide Python code that draws the chart."))
                continue  # retry

            action_type = action[0].action
            action_content = action[0].content
            logger.info("######################")
            logger.info(f"action_type: {action_type}")
            logger.info(f"action_content: {action_content}")

            if action_type != "code":
                conversation_history.append(AIMessage(content=llm_response.content))
                conversation_history.append(HumanMessage(
                    content="Your reply did not include a valid <execute> code block. Please provide Python code that draws the chart."))
                continue  # retry

            code_result = await self.code_executor.execute(code=action_content)
            logger.info(f"code_result: {code_result}")
            if code_result['error']:
                conversation_history.append(AIMessage(content=llm_response.content))
                error_feedback = (
                    "Your code failed to execute. Here is the error output:\n\n"
                    f"{code_result['stdout']}\n{code_result['stderr']}\n\nPlease fix the code and try again."
                )
                logger.info(error_feedback)
                conversation_history.append(HumanMessage(content=error_feedback))
                continue  # retry

            # Ensure the code saved a figure
            chart_filenames = re.findall(r"[\"']([^\"']+\.png)[\"']", action_content)
            if not chart_filenames:
                conversation_history.append(AIMessage(content=llm_response.content))
                feedback = "Your code ran but did not save a PNG. Please add `plt.savefig('filename.png')`."
                logger.info(feedback)
                conversation_history.append(HumanMessage(content=feedback))
                continue  # retry

            # Confirm the file exists
            potential_chart_name = os.path.basename(chart_filenames[0])
            chart_filepath = os.path.join(image_save_dir, potential_chart_name)

            if not os.path.exists(chart_filepath):
                conversation_history.append(AIMessage(content=llm_response.content))
                feedback = f"The file '{potential_chart_name}' was not found in the output directory. Please ensure the `plt.savefig()` path is correct."
                logger.info(feedback)
                conversation_history.append(HumanMessage(content=feedback))
                continue  # retry

            file_size = os.path.getsize(chart_filepath)
            if file_size > 3 * 1024 * 1024:
                conversation_history.append(AIMessage(content=llm_response.content))
                feedback = f"The file '{potential_chart_name}' was not found in the output directory. Please ensure the `plt.savefig()` path is correct."
                logger.info(feedback)
                conversation_history.append(HumanMessage(content=feedback))
                continue  # retry

            # if isinstance(llm_response, BaseMessage):
            #     llm_response = llm_response.content
            return action_content, chart_filepath

        # Bail out after three failed attempts
        return None, None

    @classmethod
    def get_provider(cls, config):
        def provider():
            agent_config = AgentConfig(
                id="chart_drawer",
                name="chart_drawer",
                description="draw charts",
                metadata={
                    "target_language": config.config["language"],
                    "working_dir": config.config["working_dir"],
                    "config": config
                }
            )

            llm_client = OpenAIClient(**config.config["llm_config_list"][0])
            vlm_client = OpenAIClient(**config.config["llm_config_list"][2])
            return cls(config=AgentRuntimeConfig(agent_config), llm_client=llm_client, vlm_client=vlm_client)
        return provider
