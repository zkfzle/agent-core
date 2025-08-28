"""
端到端（End-to-End）测试：信息抽取类任务提示词自优化
"""
import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from jiuwen.agent_builder.prompt_builder.tune.base.context_manager import ContextManager, TaskStatus
from jiuwen.agent_builder.prompt_builder.tune.base.utils import LLMModelInfo, TaskInfo, OptimizeInfo, LLMModelProcess
from jiuwen.agent_builder.prompt_builder.tune.base.case import Case
from jiuwen.agent_builder.prompt_builder.tune.joint_optimizer import JointOptimizer


API_BASE = os.getenv("API_BASE", "")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "")

# --------------------------- 待优化提示词 --------------------------- #
INFORMATION_EXTRACTION_TEMPLATE = """
你是一个信息抽取助手，请从给定句子中提取所有的人名名称
输出格式为[人名1, 人名2, ...]的列表形式，不要输出其他内容
以下是用户输入：
"""

# --------------------------- 提示词相关用例 --------------------------- #
INFORMATION_EXTRACTION_CASES = [
    Case(messages=[
        {
            "role": "user",
            "content": "潘之恒（约1536—1621）字景升，号鸾啸生，冰华生，安徽歙县、岩寺人，侨寓金陵（今江苏南京）"
        },
        {
            "role": "assistant",
            "content": "[潘之恒]"
        }
    ]),
    Case(messages=[
        {
            "role": "user",
            "content": "高祖二十二子：窦皇后生建成（李建成）、太宗皇帝（李世民）、玄霸（李玄霸）、元吉（李元吉），万贵妃生智云（李智云），莫嫔生元景（李元景），孙嫔生元昌（李元昌）"
        },
        {
            "role": "assistant",
            "content": "[李建成, 李世民, 李玄霸, 李元吉, 李智云, 李元景, 李元昌]"
        }
    ]),
    Case(messages=[
        {
            "role": "user",
            "content": "郭造卿（1532—1593），字建初，号海岳，福建福清县化南里人（今福清市人），郭遇卿之弟，郭造卿少年的时候就很有名气，曾游学吴越"
        },
        {
            "role": "assistant",
            "content": "[郭造卿, 郭遇卿]"
        }
    ]),
    Case(messages=[
        {
            "role": "user",
            "content": "沈自邠，字茂仁，号几轩，又号茂秀，浙江秀水长溪（今嘉兴南汇）人"
        },
        {
            "role": "assistant",
            "content": "[沈自邠]"
        }
    ])
]

class PromptTuneTest(unittest.TestCase):
    # ------------------------------------------------------------------ #
    #                          提示词自由化初始化方法                        #
    # ------------------------------------------------------------------ #
    def prepare_data(self):
        # 原始提示词/待优化提示词
        self.raw_prompt = INFORMATION_EXTRACTION_TEMPLATE
        # 用例信息
        self.cases = INFORMATION_EXTRACTION_CASES
        # 初始化算法优化模型信息
        self.opt_model_info = LLMModelInfo(
            url=API_BASE,
            model=MODEL_NAME,
            api_key=API_KEY,
            model_source=MODEL_PROVIDER,
        )
        # 初始化推理模型信息
        self.infer_model_info = LLMModelInfo(
            url=API_BASE,
            model=MODEL_NAME,
            api_key=API_KEY,
            model_source=MODEL_PROVIDER
        )

    def evaluate(self, prompt: str):
        llm = LLMModelProcess(self.opt_model_info)
        for case in INFORMATION_EXTRACTION_CASES:
            reply = llm.chat([
                {
                    "role": "user",
                    "content": prompt,
                },
                *(case.messages[:-1])
            ])
            print(reply.get("content").strip())

    # ------------------------------------------------------------------ #
    #                            测试用例本身                              #
    # ------------------------------------------------------------------ #
    @unittest.skip("skip system test")
    def test_information_extraction_prompt_optimization(self):
        """测试信息抽取类任务提示词优化"""
        # 步骤一. 加载原始提示词、用例
        self.prepare_data()

        # 步骤二：填写自由化任务基本信息
        task_id = "JOINT_123456"
        task_info = TaskInfo(
            task_id=task_id,
            task_name="information extraction task"
        )

        # 步骤三：填写提示词优化器超参数
        optimize_info = OptimizeInfo(
            cases=self.cases,
            num_iterations=1,
            num_parallel=5,
            user_compare_rules=""
        )
        # 步骤四：创建提示词优化器，开始优化
        optimizer = JointOptimizer()
        optimizer.do_optimize(task_info=task_info,
                              optimize_info=optimize_info,
                              raw_templates=[INFORMATION_EXTRACTION_TEMPLATE],
                              opt_model_info=self.opt_model_info,
                              infer_model_info=self.infer_model_info)


        # 步骤五：获取优化结果
        progress = ContextManager().get_task_progress(task_id)
        if progress.status == TaskStatus.TASK_STOPPED:
            print("自优化任务暂停")
            return
        print("[优化后成功率]:", progress.best_accuracy)
        print("[优化后提示词]:", progress.best_prompt)
        print("[原提示词推理效果]:")
        self.evaluate(INFORMATION_EXTRACTION_TEMPLATE)
        print("[优化后提示词推理效果]:")
        self.evaluate(progress.best_prompt)

    @unittest.skip("skip system test")
    def test_optimization_task_control(self):
        task_id = "JOINT_123456"
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.test_information_extraction_prompt_optimization)
            time.sleep(5.0)
            progress = ContextManager().get_task_progress(task_id)
            self.assertEqual(progress.status, TaskStatus.TASK_RUNNING)
            result = progress.stop()
            self.assertEqual(result, True)
            self.assertEqual(progress.status, TaskStatus.TASK_STOPPING)
            for _ in range(20):
                time.sleep(1)
                if progress.status == TaskStatus.TASK_STOPPING:
                    time.sleep(1)
                else:
                    break
            self.assertEqual(progress.status, TaskStatus.TASK_STOPPED)

        optimizer = JointOptimizer()
        print("任务重启")
        optimizer.continue_optimize(task_id)
        self.assertEqual(progress.status, TaskStatus.TASK_FINISHED)
