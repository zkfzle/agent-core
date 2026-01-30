import os.path
import pickle
import re

from openjiuwen.core.runner.runner import Runner

from src.utils import AsyncCodeExecutor


def _parse_generated_report(analysis_task, response: str):
    report_content = response
    report_title = f"{analysis_task}"

    try:
        split_report_content = report_content.split("\n")
        for idx, line in enumerate(split_report_content):
            if idx > 5:
                continue
            if line.startswith("#"):
                report_title = line.strip("#")
                break
    except Exception:
        pass
    return report_title, report_content


class DataAnalyzer:
    def __init__(self, config):
        self._state = {}
        self._code_executor_state = None
        self.config = config
        self.working_dir = config["working_dir"]

    def save(self, task_id):
        with open(os.path.join(self.working_dir, f"data_analyzer_{task_id}.pkl"), "wb") as f:
            pickle.dump(self._state, f)

        if self._code_executor_state:
            with open(os.path.join(self.working_dir, f"code_executor_{task_id}.pkl"), "wb") as f:
                pickle.dump(self._code_executor_state, f)

    def load(self, task_id):
        try:
            with open(os.path.join(self.working_dir, f"data_analyzer_{task_id}.pkl"), "rb") as f:
                self._state = pickle.load(f)
        except FileNotFoundError:
            pass
        try:
            with open(os.path.join(self.working_dir, f"code_executor_{task_id}.pkl"), "rb") as f:
                self._code_executor_state = pickle.load(f)
        except FileNotFoundError:
            pass


    async def invoke(self, task):
        inputs = task.inputs
        self.load(task_id=task.id)
        code_executor = AsyncCodeExecutor(self.working_dir)
        if self._code_executor_state:
            code_executor.load_state(self._code_executor_state)

        inputs = {
            **inputs,
            "code_executor": code_executor
        }

        report_title = self._state.get("report_title", None)
        report_content = self._state.get("report_content", None)
        if not (report_title and report_content):
            response = await Runner.run_agent("report_analyzer", inputs)
            report_title, report_content = _parse_generated_report(inputs["analysis_task"], response["final_result"])
            self._state["report_title"] = report_title
            self._state["report_content"] = report_content
            self._code_executor_state = code_executor.save_state()
            self.save(task.id)
        inputs = {
            "report_content": report_content,
            "report_title": report_title,
            **inputs
        }

        chart_code_mapping, name_mapping, name_description_mapping = await Runner.run_agent("chart_drawer", inputs)

        task.result = AnalysisResult(
            title=report_title,
            content=report_content,
            image_save_dir=os.path.join(self.working_dir, "images"),
            chart_code_mapping=chart_code_mapping,
            chart_name_mapping=name_mapping,
            chart_name_description_mapping=name_description_mapping
        )
        task.status = "finished"

class AnalysisResult:
    def __init__(
            self,
            title: str,
            content: str,
            image_save_dir: str,
            chart_code_mapping: dict = None,
            chart_name_mapping: dict = None,
            chart_name_description_mapping: dict = None
    ):
        self.title = title
        self.content = content
        self.image_save_dir = image_save_dir
        self.chart_code_mapping = chart_code_mapping
        self.chart_name_mapping = chart_name_mapping
        self.chart_name_description_mapping = chart_name_description_mapping

    def __str__(self):
        # Replace placeholders with descriptive captions
        content = self._replace_image_name()[1]
        return f"Report Title: {self.title}\nReport Content: {content}\n\n"

    def brief_str(self):
        # Replace placeholders with descriptive captions
        content = self._replace_image_name()[1]
        return f"Report Title: {self.title}\nReport Content: {content[:300]}...(more content available)\n\n"

    def _replace_image_name(self):
        image_name_list = []
        report_content = self.content
        img_list = re.findall("@import \"(.*?)\"", self.content)
        # Note: AnalysisResult is not an agent and has no logger; use prints or another mechanism if logging is needed.

        for img in img_list:
            if img in self.chart_name_description_mapping:
                new_img = self.chart_name_mapping[img]
                report_content = report_content.replace(
                    f"@import \"{img}\"",
                    f"@import \"{new_img}\"" + '\n(Description: ' + self.chart_name_description_mapping[img][:100] + ')'
                )
                image_name_list.append(new_img)
        return image_name_list, report_content

    def get_all_img(self):
        return self._replace_image_name()[0]
