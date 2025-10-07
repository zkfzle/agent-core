# -*- coding: utf-8 -*-
"""
prompt optimization evaluators
"""

import re
from typing import List, Optional

from jiuwen.agent_builder.prompt_builder.tune.utils import TuneUtils
from jiuwen.core.agent.agent import Agent
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.agent_builder.prompt_builder.tune.base import EvaluatedCase
from jiuwen.agent_builder.prompt_builder.tune.optimizer.base import BaseOptimizer



PROMPT_INSTRUCTION_OPTIMIZE_TEMPLATE = Template(content="""
你是一位提示词优化专家，你的任务是根据提供的信息对提示词进行优化。具体信息如下:
首先，请阅读以下提示词:
<prompt_base>
{{prompt_instruction}}
</prompt_base>

你拥有的的工具和API说明如下:
<tools_description>
{{tools_description}}
</tools_description>

提示词在应用的过程中出现的错误case如下：
<bad_cases>
{{bad_cases}}
</bad_cases>

对这些错误case的反思如下:
<reflections_on_bad_cases>
{{reflections_on_bad_cases}}
</reflections_on_bad_cases>

在优化提示词模版时，请遵循如下要求:
1. 在<思考>标签中，请根据错误示例及其对应的反思内容，深入、全面地分析提示词中可能导致错误的部分。分析应覆盖：错误原因的识别、原始提示词中存在的问题，以及通过哪些具体修改可以有效规避这些问题。
2. 在<PROMPT_OPTIMIZED>标签中，基于上述分析，输出优化后的提示词版本。
3. 分析过程中应聚焦于问题的具体成因，结合模板结构、语意表达和格式规范等方面，系统性地进行优化。
4. 优化过程中务必信息表达完整、逻辑侵袭，不可遗漏重要内容或引入模糊表达
5. 不可直接使用给定的示例，也不要在提示词中加入示例中的具体信息，可以通过抽象、改写的方式总结

输出格式：
<思考>
[在此详细说明你对提示词的优化分析]
</思考>
<PROMPT_OPTIMIZED>
[在此输出优化后的提示词]
</PROMPT_OPTIMIZED>
请确保优化后的内容能够有效避免之前出现的错误case。
""")

CREATE_PROMPT_TEXTUAL_GRADIENT_TEMPLATE = Template(content="""
作为提示词优化专家，我的目标是帮助代理高效且成功地完成任务
当前的提示词是:“{instruction}”
然而，这个提示词在以下实例中并未能给出正确的结果
“{{bad_cases}}”

请提供详细的反馈，分析指令可能出错的原因。
针对每个实例，具体说明指令中的问题，解释代理为何会误解指令，并提出如何让指令更加清晰和精确的建议
针对因模型调用失败导致的失败原因，可以不分析
每个反馈信息请用<INS>和</INS>包裹
""")

CREATE_BAD_CASE_TEMPLATE = Template(content="""
[question]: {{question}}
[expected answer]: {{label}}
[assistant answer]: {{answer}}
[reason]: {{reason}}
=== 
""")


class InstructionOptimizer(BaseOptimizer):
    def __init__(self,
                 agent: Agent,
                 model: BaseChatModel,
                 model_name: str,
                 **kwargs):
        super().__init__(agent)
        self._model = model
        self._model_name = model_name
        self._bad_cases_string: str = ""

    def _backward(self,
                 evaluated_cases: List[EvaluatedCase],
                 ):
        """optimize Instruction"""
        for name, param in self._parameters.items():
            textual_gradient =  self._get_textual_gradient(
                param.llm_call.get_system_prompt(), None
            )
            param.set_gradient("system_prompt", textual_gradient)

    def _update(self) -> Optional[Agent]:
        optimized_agent = self._agent.copy()
        llm_calls = optimized_agent.get_llm_calls()
        for name, param in self._parameters.items():
            optimized_prompt = self._optimize_instruction(
                param.llm_call.get_system_prompt(), param.get_gradient("system_prompt"), None
            )
            param.llm_call.update_system_prompt(optimized_prompt)
            llm_calls.get(name).update_system_prompt(optimized_prompt)
        return optimized_agent

    def _get_textual_gradient(self,
                              original_prompt: Template,
                              tools: Optional[list] = None) -> str:
        prompt = TuneUtils.get_content_string_from_template(original_prompt)
        messages = CREATE_PROMPT_TEXTUAL_GRADIENT_TEMPLATE.format(
            dict(instruction=prompt, bad_cases=self._get_bad_cases_string())
        ).to_messages()
        textual_gradient = self._model.invoke(self._model_name, messages).content
        return textual_gradient

    def _optimize_instruction(self,
                              instruction: Template,
                              textual_gradient,
                              tools):
        """update instruction"""
        messages = PROMPT_INSTRUCTION_OPTIMIZE_TEMPLATE.format(
            dict(prompt_instruction=TuneUtils.get_content_string_from_template(instruction),
                 bad_cases=self._get_bad_cases_string(),
                 reflections_on_bad_cases=textual_gradient,
                 tools_description=str(tools) if tools else "None"
                 )
        ).to_messages()
        response = self._model.invoke(self._model_name, messages).content
        return self._extract_optimized_prompt_from_response(response)

    @staticmethod
    def _extract_optimized_prompt_from_response(response) -> Optional[str]:
        """extract optimized prompt from response"""
        optimized_prompt_pattern = r"<PROMPT_OPTIMIZED>(.*?)</PROMPT_OPTIMIZED>"
        match = re.search(optimized_prompt_pattern, response, re.DOTALL)
        if not match:
            return None
        optimized_prompt = match.group(1)
        return optimized_prompt.replace("<prompt_base>", "").replace("</prompt_base>", "")

    def _get_bad_cases_string(self) -> str:
        error_example_string = "\n".join(
            CREATE_BAD_CASE_TEMPLATE.format(
                dict(question=str(eval_case.case.inputs),
                     label=str(eval_case.case.label),
                     answer=str(eval_case.answer),
                     reason=eval_case.reason)
            ).content
            for eval_case in self._bad_cases
        )
        return error_example_string