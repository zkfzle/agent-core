import unittest

from jiuwen.core.context.model_context.model_context import ModelContext
from jiuwen.core.context.model_context.config import ModelContextConfig
from jiuwen.core.context_engine.base import EngineInput, EngineOutput
from jiuwen.core.context_engine.processor.assemble.base import AssembleStage
from jiuwen.core.context_engine.processor.preprocess.base import PreprocessStage
from jiuwen.core.context_engine.processor.postprocess.base import PostprocessStage
from jiuwen.core.context_engine.processor.factory import ProcessorFactory
from jiuwen.core.context_engine.config import BaseProcessorConfig, ContextEngineConfig, OnlineExecuteConfig

"""preprocessor"""
class TestMemoryCompressorConfig(BaseProcessorConfig):
    processor_type: str = "test_memory_compressor"

@ProcessorFactory.register("test_memory_compressor", TestMemoryCompressorConfig)
class TestMemoryCompressor(PreprocessStage):
    def __init__(self, config: TestMemoryCompressorConfig):
        super().__init__(config)

    def run(self, input_data: EngineInput) -> EngineOutput:
        output: EngineOutput = EngineOutput.from_input(input_data)
        if len(input_data.chat_history) > 2:
            output.chat_history = input_data.chat_history[-2:]
        return output


"""assembler"""
class TestAssemblerConfig(BaseProcessorConfig):
    processor_type: str = "test_assembler"


@ProcessorFactory.register("test_assembler", TestAssemblerConfig)
class TestAssembler(AssembleStage):
    def __init__(self, config: TestAssemblerConfig):
        super().__init__(config)

    def run(self, input_data: EngineInput) -> EngineOutput:
        from jiuwen.core.utils.prompt.assemble.assembler import Assembler
        assembler = Assembler(input_data.system_prompt, return_format="text")
        result = assembler.assemble(**input_data.user_variables)
        history_str = '\n'.join([f"[{msg.role}]:{msg.content}" for msg in input_data.chat_history])
        result += f"\nhistory:\n{history_str}\n"
        result += f"query: {input_data.user_input}\n"
        output = EngineOutput.from_input(input_data)
        output.full_output = result
        return output


"""postprocessor"""
class TestSuffixAdderConfig(BaseProcessorConfig):
    processor_type: str = "test_suffix_adder"


@ProcessorFactory.register("test_suffix_adder", TestSuffixAdderConfig)
class TestSuffixAdder(PostprocessStage):
    def __init__(self, config: TestSuffixAdderConfig):
        super().__init__(config)

    def run(self, input_data: EngineInput) -> EngineOutput:
        output = EngineOutput.from_input(input_data)
        output.full_output = input_data.full_output + "\n这是后缀信息"
        return output


class ContextEngineTest(unittest.TestCase):
    def test_online_pipeline(self):
        """test_online_pipeline"""
        config = ModelContextConfig(
            engine_config=ContextEngineConfig(
                node_id="123",
                online_process=OnlineExecuteConfig(
                    preprocess_stage=[
                        TestMemoryCompressorConfig()
                    ],
                    assemble_stage=[
                        TestAssemblerConfig()
                    ],
                    postprocess_stage=[
                        TestSuffixAdderConfig()
                    ]
                )
            )
        )

        context = ModelContext("12345", config)
        context.add_user_message("世界上最高的山是什么")
        context.add_assistant_message("世界上最高的山是珠穆朗玛峰")
        context.add_user_message("世界上最长的河是什么")
        context.add_assistant_message("世界上最长的河是亚马逊河")
        system_prompt = "你是一个{{role}}小助手，请根据指定用户回答完成指定任务。"
        user_query = "世界上最深的湖是什么？"
        result = context.process(user_query, system_prompt)
        print("处理后输出：", result.full_output)

