import unittest

from jiuwen.core.context_engine.base import ContextWindow, ContextVariable
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.context_engine.processor.base import BaseContextProcessor
from jiuwen.core.context_engine.processor.factory import ProcessorFactory
from jiuwen.core.context_engine.config import BaseProcessorConfig, ContextEngineConfig
from jiuwen.core.context_engine.processor.asynch.variable_extractor import VariableExtractorConfig
from jiuwen.core.utils.llm.messages import HumanMessage, AIMessage


"""preprocessor"""
class TestMemoryCompressorConfig(BaseProcessorConfig):
    processor_type: str = "test_memory_compressor"


@ProcessorFactory.register("test_memory_compressor", TestMemoryCompressorConfig)
class TestMemoryCompressor(BaseContextProcessor):
    def __init__(self, config: TestMemoryCompressorConfig):
        super().__init__(config)

    def run(self, input_data: ContextWindow) -> ContextWindow:
        output: ContextWindow = input_data
        if len(input_data.chat_history) > 2:
            output.chat_history = input_data.chat_history[-2:]
        return output


"""assembler"""
class TestAssemblerConfig(BaseProcessorConfig):
    processor_type: str = "test_assembler"


@ProcessorFactory.register("test_assembler", TestAssemblerConfig)
class TestAssembler(BaseContextProcessor):
    def __init__(self, config: TestAssemblerConfig):
        super().__init__(config)

    def run(self, input_data: ContextWindow) -> ContextWindow:
        from jiuwen.core.utils.prompt.assemble.assembler import Assembler
        assembler = Assembler(input_data.system_prompt, return_format="text")
        variables = dict([(name, var.value) for name, var in input_data.variables.items()])
        result = assembler.assemble(**variables)
        history_str = '\n'.join([f"[{msg.role}]:{msg.content}" for msg in input_data.chat_history])
        result += f"\nhistory:\n{history_str}\n"
        result += f"query: {input_data.user_input}\n"
        output = input_data
        output.full_prompt = result
        return output


"""postprocessor"""
class TestSuffixAdderConfig(BaseProcessorConfig):
    processor_type: str = "test_suffix_adder"


@ProcessorFactory.register("test_suffix_adder", TestSuffixAdderConfig)
class TestSuffixAdder(BaseContextProcessor):
    def __init__(self, config: TestSuffixAdderConfig):
        super().__init__(config)

    def run(self, input_data: ContextWindow) -> ContextWindow:
        output = input_data
        output.full_prompt = input_data.full_prompt + "这是后缀信息"
        return output


class ContextEngineTest(unittest.TestCase):
    def test_context_processing_pipeline(self):
        """test_online_pipeline"""
        config = ContextEngineConfig(
            variables=[
                ContextVariable(name="最高的山", default_value=""),
                ContextVariable(name="最长的河", default_value=""),
                ContextVariable(name="最深的湖", default_value=""),
            ],
            processors=[
                TestMemoryCompressorConfig(),
                TestAssemblerConfig(),
                TestSuffixAdderConfig()
            ],
            async_processors=[
                VariableExtractorConfig(processor_type="variable_extractor"),
            ]
        )

        ce_engine = ContextEngine("123", config)
        context = ce_engine.get_agent_context(session_id="456")
        context.add_message(HumanMessage(content="世界上最高的山是什么"))
        context.add_message(AIMessage(content="世界上最高的山是珠穆朗玛峰"))
        context.add_message(HumanMessage(content="世界上最长的河是什么"))
        context.add_message(AIMessage(content="世界上最长的河是尼罗河"))
        system_prompt = "你是一个{{role}}小助手，请根据指定用户回答完成指定任务。"
        user_query = "世界上最深的湖是什么？"
        result = context.assemble_by_pipeline(user_query, system_prompt,
                                              variables={
                                                  "role": ContextVariable(name="role", value="问答")
                                              })
        expected_result = "你是一个问答小助手，请根据指定用户回答完成指定任务。\n" \
                          "history:\n" \
                          "[user]:世界上最长的河是什么\n" \
                          "[assistant]:世界上最长的河是尼罗河\n" \
                          "query: 世界上最深的湖是什么？\n" \
                          "这是后缀信息"
        self.assertEqual(result, expected_result)
        result = context.assemble(user_query, system_prompt,
                                  variables={
                                      "role": ContextVariable(name="role", value="问答"),
                                  })
        expected_result = "你是一个问答小助手，请根据指定用户回答完成指定任务。\n" \
                          "history:\n" \
                          "[user]:世界上最高的山是什么\n" \
                          "[assistant]:世界上最高的山是珠穆朗玛峰\n" \
                          "[user]:世界上最长的河是什么\n" \
                          "[assistant]:世界上最长的河是尼罗河\n" \
                          "query: 世界上最深的湖是什么？"
        self.assertEqual(result, expected_result)
