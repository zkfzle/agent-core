import unittest

from jiuwen.core.utils.llm.messages import SystemMessage
from jiuwen.core.context_engine.base import ContextWindow, ContextVariable
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.context_engine.processor.assemble.assembler import AssemblerConfig
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
        assembler = Assembler(input_data.prompt.content)
        variables = dict([(name, var.value) for name, var in input_data.variables.items()])
        result = assembler.assemble(**variables)
        history_str = '\n'.join([f"[{msg.role}]:{msg.content}" for msg in input_data.chat_history])
        result += f"\nhistory:\n{history_str}\n"
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
        system_prompt = "你是一个{{role}}小助手，请根据指定用户回答完成指定任务。{{user_input}}"
        user_query = "世界上最深的湖是什么？"
        result = context.assemble_by_pipeline(system_prompt,
                                              variables={
                                                  "role": "问答",
                                                  "user_input": user_query
                                              })
        expected_result = "你是一个问答小助手，请根据指定用户回答完成指定任务。世界上最深的湖是什么？\n" \
                          "history:\n" \
                          "[user]:世界上最长的河是什么\n" \
                          "[assistant]:世界上最长的河是尼罗河\n" \
                          "这是后缀信息"
        self.assertEqual(result, expected_result)
        result = context.assemble(system_prompt,
                                  variables={
                                      "role": "问答",
                                      "user_input": user_query
                                  })
        expected_result = "你是一个问答小助手，请根据指定用户回答完成指定任务。世界上最深的湖是什么？"
        self.assertEqual(result, expected_result)

        system_prompt = SystemMessage(content="你是一个{{role}}小助手，请根据指定用户回答完成指定任务。{{user_input}}")
        result = context.assemble(system_prompt,
                                  variables={
                                      "role": "问答",
                                      "user_input": user_query
                                  })
        expected_result = SystemMessage(content="你是一个问答小助手，请根据指定用户回答完成指定任务。世界上最深的湖是什么？")
        self.assertEqual(result, expected_result)

        system_prompt = [
            SystemMessage(content="你是一个{{role}}小助手，请根据指定用户回答完成指定任务。{{user_input}}"),
            HumanMessage(content="请用{{style}}的风格回答用户")
        ]
        result = context.assemble(system_prompt,
                                  variables={
                                      "role": "问答",
                                      "user_input": user_query,
                                      "style": "简洁、礼貌"
                                  })
        expected_result_1 = SystemMessage(content="你是一个问答小助手，请根据指定用户回答完成指定任务。世界上最深的湖是什么？")
        expected_result_2 = HumanMessage(content="请用简洁、礼貌的风格回答用户")
        self.assertEqual(result[0], expected_result_1)
        self.assertEqual(result[1], expected_result_2)

    def test_context_message_process(self):
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

        src_messages = [
            HumanMessage(content="世界上最高的山是什么"),
            AIMessage(content="世界上最高的山是珠穆朗玛峰"),
            HumanMessage(content="世界上最长的河是什么"),
            AIMessage(content="世界上最长的河是尼罗河")
        ]

        # get message before add
        self.assertEqual(context.get_messages(), [])
        self.assertEqual(context.get_latest_message(), None)

        # add message & get message
        context.add_message(src_messages[0])
        context.add_message(src_messages[1], tags={"label": "ai"})
        msgs1 = context.get_messages()
        self.assertEqual(len(msgs1), 2)
        self.assertEqual(msgs1, src_messages[:2])

        msgs2 = context.get_messages(tags={"label": "ai"})
        self.assertEqual(len(msgs2), 1)
        self.assertEqual(msgs2, [src_messages[1]])

        msgs3 = context.get_messages(tags={"label": "None"})
        self.assertEqual(msgs3, [])

        # batch add messages
        context.batch_add_messages(src_messages)
        msgs4 = context.get_messages()
        self.assertEqual(len(msgs4), 6)

        # get latest message
        latest_msg1 = context.get_latest_message()
        self.assertEqual(latest_msg1, src_messages[-1])

        latest_msg2 = context.get_latest_message(role="user")
        self.assertEqual(latest_msg2, src_messages[-2])
