"""
端到端（End-to-End）测试：信息抽取类任务提示词自优化
"""
import os
import unittest
from typing import List

from jiuwen.agent_builder.prompt_builder.tune.task.task import PromptTask
from jiuwen.agent_builder.prompt_builder.tune.optimizer.example_optimizer import ExampleOptimizer
from jiuwen.agent_builder.prompt_builder.tune.optimizer.instruction_optimizer import InstructionOptimizer
from jiuwen.agent_builder.prompt_builder.tune.optimizer.joint_optimizer import JointOptimizer
from jiuwen.agent_builder.prompt_builder.tune.evaluator.evaluator import DefaultEvaluator
from jiuwen.core.utils.llm.messages import AIMessage, Function, Parameters, UsageMetadata
from jiuwen.agent_builder.prompt_builder.tune.base import Case
from jiuwen.agent_builder.prompt_builder.tune.prompt_trainer import PromptTrainer
from jiuwen.agent_builder.prompt_builder.tune.dataset.case_loader import CaseLoader
from jiuwen.core.utils.llm.messages import HumanMessage, ToolInfo, ToolCall, FunctionInfo
from jiuwen.core.utils.llm.model_utils.model_factory import ModelFactory


API_BASE = os.getenv("API_BASE", "")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "")


# ——————————————————————————————————————————工具信息————————————————————————————————————#
TOOLS = [
    ToolInfo(
        function=Function(
            name="ac_open",
            description="空调控制工具，根据用户指令打开空调",
            parameters=Parameters(
                properties={},
                required=[]
            )
        )
    ),
    ToolInfo(
        function=Function(
            name="ac_close",
            description="空调控制工具，根据用户指令关闭空调",
            parameters=Parameters(
                properties={},
                required=[]
            )
        )
    ),
    ToolInfo(
        function=Function(
            name="ac_control",
            description="空调温度调节工具，按用户指令设置温度",
            parameters=Parameters(
                properties={
                    'temperature':{'description':'需要设置的温度', 'type':'int'},
                },
                required=['temperature'],
            )
        )
    ),
]
# --------------------------- 待优化提示词 --------------------------- #
INFORMATION_EXTRACTION_TEMPLATE = """
你是一个信息抽取助手，请从给定句子中提取所有的人名名称
输出格式为[人名1, 人名2, ...]的列表形式，不要输出其他内容
以下是用户输入：
"""

TOOL_CALLS_TEMPLATE = """
你是一个工具调用助手，请根据用户的指令，调用工具
"""

INFORMATION_EXTRACTION_TEMPLATE_WITH_VARIABLES = """
你是一个{{role}}助手，请从给定句子中提取所有的人名名称
输出格式为[人名1, 人名2, ...]的列表形式，不要输出其他内容
以下是用户输入：
{{user_input}}
"""

# --------------------------- 提示词相关用例 --------------------------- #
INFORMATION_EXTRACTION_CASES = [
    Case(messages=[HumanMessage(content="潘之恒（约1536—1621）字景升，号鸾啸生，冰华生，安徽歙县、岩寺人，侨寓金陵（今江苏南京）")],
        label=AIMessage(content="[潘之恒]")),
    Case(messages=[HumanMessage(content="高祖二十二子：窦皇后生建成（李建成）、太宗皇帝（李世民）、玄霸（李玄霸）、元吉（李元吉），万贵妃生智云（李智云），莫嫔生元景（李元景），孙嫔生元昌（李元昌））")],
         label=AIMessage(content="[李建成, 李世民, 李玄霸, 李元吉, 李智云, 李元景, 李元昌]")),
    Case(messages=[HumanMessage(content="郭造卿（1532—1593），字建初，号海岳，福建福清县化南里人（今福清市人），郭遇卿之弟，郭造卿少年的时候就很有名气，曾游学吴越")],
         label=AIMessage(content="[郭造卿, 郭遇卿]")),
    Case(messages=[HumanMessage(content="沈自邠，字茂仁，号几轩，又号茂秀，浙江秀水长溪（今嘉兴南汇）人")],
        label=AIMessage(content="[沈自邠]"))
]

TOOL_CALL_CASES = [
    Case(messages=[HumanMessage(content="请帮我打开空调")],
        label=AIMessage(content="", tool_calls=[
            ToolCall(args={}, id="", index=0, type='function', function=FunctionInfo(name="ac_open", arguments="{}"))],
                        usage_metadata=UsageMetadata(finish_reason="tool_calls")),
        tools=TOOLS
    ),
    Case(messages=[HumanMessage(content="请帮我关闭空调")],
         label=AIMessage(content="", tool_calls=[
             ToolCall(args={}, id="", index=0, type='function', function=FunctionInfo(name="ac_close", arguments="{}"))],
                         usage_metadata=UsageMetadata(finish_reason="tool_calls")),
         tools=TOOLS
         ),
    Case(messages=[HumanMessage(content="天气太热了，开一下空调")],
         label=AIMessage(content="", tool_calls=[
             ToolCall(args={}, id="", index=0, type='function', function=FunctionInfo(name="ac_open", arguments="{}"))],
                         usage_metadata=UsageMetadata(finish_reason="tool_calls")),
         tools=TOOLS
         ),
    Case(messages=[HumanMessage(content="有点冷，先帮我关窗，再调整到21度")],
         label=AIMessage(content="", tool_calls=[
             ToolCall(args={}, id="", index=0, type='function', function=FunctionInfo(name="ac_control", arguments="{\"temperature\":21}"))],
                         usage_metadata=UsageMetadata(finnish_reson="tool_calls")),
         tools=TOOLS
         ),
    Case(messages=[HumanMessage(content="有点热，先帮我开窗，再调整到29度")],
         label=AIMessage(content="", tool_calls=[
             ToolCall(args={}, id="", index=0, type='function', function=FunctionInfo(name="ac_control", arguments="{\"temperature\":29}"))],
                         usage_metadata=UsageMetadata(finish_reason="tool_calls")),
         tools=TOOLS
         )
]

INFORMATION_EXTRACTION_CASES_WITH_VARIABLES = [
    Case(messages=[],
         variables={
             "role":"信息提取",
             "user_input":"潘之恒（约1536—1621）字景升，号鸾啸生，冰华生，安徽歙县、岩寺人，侨寓金陵（今江苏南京）"
         },
        label=AIMessage(content="[潘之恒]")),
    Case(messages=[],
         variables={
             "role": "信息提取",
             "user_input": "高祖二十二子：窦皇后生建成（李建成）、太宗皇帝（李世民）、玄霸（李玄霸）、元吉（李元吉），万贵妃生智云（李智云），莫嫔生元景（李元景），孙嫔生元昌（李元昌））"
         },
         label=AIMessage(content="[李建成, 李世民, 李玄霸, 李元吉, 李智云, 李元景, 李元昌]")),
    Case(messages=[],
         variables={
             "role": "信息提取",
             "user_input": "郭造卿（1532—1593），字建初，号海岳，福建福清县化南里人（今福清市人），郭遇卿之弟，郭造卿少年的时候就很有名气，曾游学吴越"
         },
         label=AIMessage(content="[郭造卿, 郭遇卿]")),
    Case(messages=[],
         variables={
             "role": "信息提取",
             "user_input": "沈自邠，字茂仁，号几轩，又号茂秀，浙江秀水长溪（今嘉兴南汇）人"
         },
         label=AIMessage(content="[沈自邠]")),
]


class PromptTuneTest(unittest.TestCase):
    # ------------------------------------------------------------------ #
    #                          提示词自由化初始化方法                        #
    # ------------------------------------------------------------------ #
    def create_prompt_optimization_trainer(self, original_prompt: str, cases: List[Case]):
        # 步骤一，创建推理模型、优化模型
        infer_model = ModelFactory().get_model(
            model_provider=MODEL_PROVIDER,
            api_key=API_KEY,
            api_base=API_BASE,
        )

        opt_model = ModelFactory().get_model(
            model_provider=MODEL_PROVIDER,
            api_key=API_KEY,
            api_base=API_BASE,
        )

        #步骤二，创建提示词优化任务，注册模型+待优化提示词
        task = PromptTask(
            model=infer_model,
            model_name=MODEL_NAME,
            prompt=original_prompt,
        )
        # 步骤三: 创建数据加载器
        case_loader =CaseLoader(cases=cases)

        #步骤四:创建示例-指令联合优化器
        optimizer= JointOptimizer(
            model=opt_model,
            model_name=MODEL_NAME,
            instruction_optimizer=InstructionOptimizer(
                model_name = MODEL_NAME,
                model = opt_model
            ),
            example_optimizer=ExampleOptimizer(
                model=opt_model,
                model_name = MODEL_NAME,
                num_examples = 1,
            ),
        )

        # 步骤五:创建评估器，定义比较器和比较规则
        evaluator = DefaultEvaluator(
            model=opt_model,
            model_name=MODEL_NAME,
            metric="两个回答需要完全一致，包括数量和名字"
        )

        #步骤六:创建提示词调优器，启动调优
        trainer = PromptTrainer(
            task=task,
            optimizer=optimizer._example_optimizer,
            evaluator=evaluator,
            case_loader=case_loader,
        )
        self._trainer = trainer
        self._task = task

    def train_and_evaluate(self, original_prompt, cases):
        # 步骤七:获取优化结果
        progress = self._trainer.train(num_iterations=2)
        print("[优化后成功率]:",progress.best_accuracy)
        print("[优化后提不词]:",progress.best_prompt)
        print("[原提示词推理效果]:")
        self._task.update_prompt(original_prompt)
        accuracy, result = self._trainer.evaluate(cases)
        for eval_result in result:
            print(f"score: {eval_result.score}, reason: {eval_result.reason}, output: {eval_result.answer.content}")
        print("[优化后提示词推理效果]:")
        self._task.update_prompt(progress.best_prompt)
        accuracy, result = self._trainer.evaluate(cases)
        for eval_result in result:
            print(f"score: {eval_result.score}, reason: {eval_result.reason}, output: {eval_result.answer.content}")
        for hist in progress.history:
            print(hist.accuracy)

    # ------------------------------------------------------------------ #
    #                            测试用例本身                              #
    # ------------------------------------------------------------------ #
    @unittest.skip("skip system test")
    def test_information_extraction_prompt_optimization(self):
        self.create_prompt_optimization_trainer(
           original_prompt=INFORMATION_EXTRACTION_TEMPLATE,
           cases=INFORMATION_EXTRACTION_CASES
        )
        self.train_and_evaluate(
           original_prompt=INFORMATION_EXTRACTION_TEMPLATE,
           cases=INFORMATION_EXTRACTION_CASES
        )

    @unittest.skip("skip system test")
    def test_tool_calls_prompt_optimization(self):
        self.create_prompt_optimization_trainer(
           original_prompt=TOOL_CALLS_TEMPLATE,
           cases=TOOL_CALL_CASES
        )
        self.train_and_evaluate(
           original_prompt=TOOL_CALLS_TEMPLATE,
           cases=TOOL_CALL_CASES
        )

    @unittest.skip("skip system test")
    def test_information_extraction_prompt_optimization_with_variables(self):
        self.create_prompt_optimization_trainer(
           original_prompt=INFORMATION_EXTRACTION_TEMPLATE_WITH_VARIABLES,
           cases=INFORMATION_EXTRACTION_CASES_WITH_VARIABLES
        )
        self.train_and_evaluate(
           original_prompt=INFORMATION_EXTRACTION_TEMPLATE_WITH_VARIABLES,
           cases=INFORMATION_EXTRACTION_CASES_WITH_VARIABLES
        )