#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
import unittest
from unittest.mock import MagicMock, patch

from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.common.exception.exception import JiuWenBaseException
import jiuwen.agent_builder.nl_to_agent.agent_builder.workflow_builder.workflow_builder as wb
from jiuwen.agent_builder.nl_to_agent.agent_builder.workflow_builder.workflow_builder import WorkflowBuilder, State


TRANSFORMED_SOP = "transformed_sop"
GENERATED_SOP = "generated_sop"
GENERATED_DL = "generated_dl"
REFINED_DL = "refined_dl"
MERMAID = "mermaid"
DSL = "dsl"


class TestWorkflowBuilder(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock(name="llm")
        self.context_manager = MagicMock(name="context_manager")
        self.context_manager.get_filtered_messages.return_value = []

        self.mock_intention_detector = MagicMock(name="IntentionDetector")

        self.mock_sop_generator = MagicMock(name="SopGenerator")
        self.mock_sop_generator.transform.return_value = TRANSFORMED_SOP
        self.mock_sop_generator.generate.return_value = GENERATED_SOP

        self.mock_dl_generator = MagicMock(name="DLGenerator")
        self.mock_dl_generator.generate.return_value = GENERATED_DL
        self.mock_dl_generator.refine.return_value = REFINED_DL
        self.mock_dl_generator.reflect_prompts = []

        self.mock_reflector = MagicMock(name="Reflector")
        self.mock_reflector.errors = []

        self.mock_transformer = MagicMock(name="DLTransformer")
        self.mock_transformer.transform_to_mermaid.return_value = MERMAID
        self.mock_transformer.transform_to_dsl.return_value = DSL

        self.mock_resource_retriever = MagicMock(name="ResourceRetriever")
        self.mock_resource_retriever.retrieve.return_value = {"res": "value"}

    def test_init_builder(self):
        with patch.object(wb, 'IntentionDetector', side_effect=lambda llm: self.mock_intention_detector), \
             patch.object(wb, 'SopGenerator', side_effect=lambda llm, ctx: self.mock_sop_generator), \
             patch.object(wb, 'ResourceRetriever', side_effect=lambda: self.mock_resource_retriever), \
             patch.object(wb, 'DLGenerator', side_effect=lambda llm, ctx: self.mock_dl_generator), \
             patch.object(wb, 'Reflector', side_effect=lambda: self.mock_reflector), \
             patch.object(wb, 'DLTransformer', side_effect=lambda llm, ctx: self.mock_transformer):
            builder = WorkflowBuilder(self.llm, self.context_manager)

        self.assertEqual(builder._state, State.INITIAL)
        self.assertIsNone(builder._dl)
        self.assertIsNone(builder._mermaid_code)
        self.assertIsNone(builder._resource)

    def test_invalid_state(self):
        with patch.object(wb, 'IntentionDetector', side_effect=lambda llm: self.mock_intention_detector), \
             patch.object(wb, 'SopGenerator', side_effect=lambda llm, ctx: self.mock_sop_generator), \
             patch.object(wb, 'ResourceRetriever', side_effect=lambda: self.mock_resource_retriever), \
             patch.object(wb, 'DLGenerator', side_effect=lambda llm, ctx: self.mock_dl_generator), \
             patch.object(wb, 'Reflector', side_effect=lambda: self.mock_reflector), \
             patch.object(wb, 'DLTransformer', side_effect=lambda llm, ctx: self.mock_transformer):
            builder = WorkflowBuilder(self.llm, self.context_manager)
            builder._state = 'test'

        with self.assertRaises(JiuWenBaseException) as cm:
            result = builder.execute("query")
        self.assertIn(str(StatusCode.NL2AGENT_WORKFLOW_STATE_ERROR.code), str(cm.exception))
        self.assertIn("未知的工作流阶段", str(cm.exception))

    def test_initial_without_provide_process(self):
        self.mock_intention_detector.detect_initial_instruction.return_value = False
        with patch.object(wb, 'IntentionDetector', side_effect=lambda llm: self.mock_intention_detector), \
             patch.object(wb, 'SopGenerator', side_effect=lambda llm, ctx: self.mock_sop_generator), \
             patch.object(wb, 'ResourceRetriever', side_effect=lambda: self.mock_resource_retriever), \
             patch.object(wb, 'DLGenerator', side_effect=lambda llm, ctx: self.mock_dl_generator), \
             patch.object(wb, 'Reflector', side_effect=lambda: self.mock_reflector), \
             patch.object(wb, 'DLTransformer', side_effect=lambda llm, ctx: self.mock_transformer):
            builder = WorkflowBuilder(self.llm, self.context_manager)
            result = builder.execute("query without workflow description")

        self.assertEqual(result, wb.WORKFLOW_REQUEST_CONTENT)
        self.assertEqual(builder._state, State.PROCESS_REQUEST)

    def test_initial_with_provide_process(self):
        self.mock_intention_detector.detect_initial_instruction.return_value = True
        with patch.object(wb, 'IntentionDetector', side_effect=lambda llm: self.mock_intention_detector), \
             patch.object(wb, 'SopGenerator', side_effect=lambda llm, ctx: self.mock_sop_generator), \
             patch.object(wb, 'ResourceRetriever', side_effect=lambda: self.mock_resource_retriever), \
             patch.object(wb, 'DLGenerator', side_effect=lambda llm, ctx: self.mock_dl_generator), \
             patch.object(wb, 'Reflector', side_effect=lambda: self.mock_reflector), \
             patch.object(wb, 'DLTransformer', side_effect=lambda llm, ctx: self.mock_transformer):
            builder = WorkflowBuilder(self.llm, self.context_manager)
            result = builder.execute("query with workflow description")

        self.assertEqual(result, MERMAID)
        self.assertEqual(builder._state, State.PROCESS_CONFIRM)
        self.assertEqual(builder._dl, GENERATED_DL)
        self.assertEqual(builder._resource, {"res": "value"})
        self.mock_sop_generator.transform.assert_called()
        self.mock_reflector.check_format.assert_called()

    def test_process_request_without_provide_process(self):
        self.mock_intention_detector.detect_initial_instruction.return_value = False
        with patch.object(wb, 'IntentionDetector', side_effect=lambda llm: self.mock_intention_detector), \
             patch.object(wb, 'SopGenerator', side_effect=lambda llm, ctx: self.mock_sop_generator), \
             patch.object(wb, 'ResourceRetriever', side_effect=lambda: self.mock_resource_retriever), \
             patch.object(wb, 'DLGenerator', side_effect=lambda llm, ctx: self.mock_dl_generator), \
             patch.object(wb, 'Reflector', side_effect=lambda: self.mock_reflector), \
             patch.object(wb, 'DLTransformer', side_effect=lambda llm, ctx: self.mock_transformer):
            builder = WorkflowBuilder(self.llm, self.context_manager)
            builder._state = State.PROCESS_REQUEST
            result = builder.execute("query without workflow description")

        self.assertEqual(result, MERMAID)
        self.assertEqual(builder._state, State.PROCESS_CONFIRM)
        self.assertEqual(builder._dl, GENERATED_DL)
        self.assertEqual(builder._resource, {"res": "value"})
        self.mock_sop_generator.generate.assert_called()
        self.mock_reflector.check_format.assert_called()
            
    def test_process_request_with_provide_process(self):
        self.mock_intention_detector.detect_initial_instruction.return_value = True
        with patch.object(wb, 'IntentionDetector', side_effect=lambda llm: self.mock_intention_detector), \
             patch.object(wb, 'SopGenerator', side_effect=lambda llm, ctx: self.mock_sop_generator), \
             patch.object(wb, 'ResourceRetriever', side_effect=lambda: self.mock_resource_retriever), \
             patch.object(wb, 'DLGenerator', side_effect=lambda llm, ctx: self.mock_dl_generator), \
             patch.object(wb, 'Reflector', side_effect=lambda: self.mock_reflector), \
             patch.object(wb, 'DLTransformer', side_effect=lambda llm, ctx: self.mock_transformer):
            builder = WorkflowBuilder(self.llm, self.context_manager)
            builder._state = State.PROCESS_REQUEST
            result = builder.execute("query without workflow description")

        self.assertEqual(result, MERMAID)
        self.assertEqual(builder._state, State.PROCESS_CONFIRM)
        self.assertEqual(builder._dl, GENERATED_DL)
        self.assertEqual(builder._resource, {"res": "value"})
        self.mock_sop_generator.transform.assert_called()
        self.mock_reflector.check_format.assert_called()
            
    def test_process_confirm_with_refine(self):
        self.mock_intention_detector.detect_refine_intent.return_value = True
        with patch.object(wb, 'IntentionDetector', side_effect=lambda llm: self.mock_intention_detector), \
             patch.object(wb, 'SopGenerator', side_effect=lambda llm, ctx: self.mock_sop_generator), \
             patch.object(wb, 'ResourceRetriever', side_effect=lambda: self.mock_resource_retriever), \
             patch.object(wb, 'DLGenerator', side_effect=lambda llm, ctx: self.mock_dl_generator), \
             patch.object(wb, 'Reflector', side_effect=lambda: self.mock_reflector), \
             patch.object(wb, 'DLTransformer', side_effect=lambda llm, ctx: self.mock_transformer):
            builder = WorkflowBuilder(self.llm, self.context_manager)
            builder._state = State.PROCESS_CONFIRM
            result = builder.execute("query with refine")

        self.assertEqual(result, MERMAID)
        self.assertEqual(builder._state, State.PROCESS_CONFIRM)
        self.assertEqual(builder._dl, REFINED_DL)
        self.mock_reflector.check_format.assert_called()
            
    def test_process_confirm_without_refine(self):
        self.mock_intention_detector.detect_refine_intent.return_value = False
        with patch.object(wb, 'IntentionDetector', side_effect=lambda llm: self.mock_intention_detector), \
             patch.object(wb, 'SopGenerator', side_effect=lambda llm, ctx: self.mock_sop_generator), \
             patch.object(wb, 'ResourceRetriever', side_effect=lambda: self.mock_resource_retriever), \
             patch.object(wb, 'DLGenerator', side_effect=lambda llm, ctx: self.mock_dl_generator), \
             patch.object(wb, 'Reflector', side_effect=lambda: self.mock_reflector), \
             patch.object(wb, 'DLTransformer', side_effect=lambda llm, ctx: self.mock_transformer):
            builder = WorkflowBuilder(self.llm, self.context_manager)
            builder._state = State.PROCESS_CONFIRM
            result = builder.execute("query without refine")

        self.assertEqual(result, DSL)
        self.assertEqual(builder._state, State.INITIAL)
        self.assertIsNone(builder._dl)
        self.assertIsNone(builder._mermaid_code)
        self.assertIsNone(builder._resource)
        self.assertEqual(len(builder._dl_generator.reflect_prompts), 0)

    def test_generate_dl_success(self):
        with patch.object(wb, 'IntentionDetector', side_effect=lambda llm: self.mock_intention_detector), \
             patch.object(wb, 'SopGenerator', side_effect=lambda llm, ctx: self.mock_sop_generator), \
             patch.object(wb, 'ResourceRetriever', side_effect=lambda: self.mock_resource_retriever), \
             patch.object(wb, 'DLGenerator', side_effect=lambda llm, ctx: self.mock_dl_generator), \
             patch.object(wb, 'Reflector', side_effect=lambda: self.mock_reflector), \
             patch.object(wb, 'DLTransformer', side_effect=lambda llm, ctx: self.mock_transformer):
            builder = WorkflowBuilder(self.llm, self.context_manager)

        generated_dl = builder._generate_and_reflect_dl(
            dl_operation=builder._dl_generator.generate,
            query=wb.GENERATE_DL_FROM_SOP_CONTENT + TRANSFORMED_SOP,
            resource=None
        )
        self.assertEqual(self.mock_dl_generator.generate.call_count, 1)
        self.mock_dl_generator.generate.assert_called_with(
            query=wb.GENERATE_DL_FROM_SOP_CONTENT + TRANSFORMED_SOP, resource=None
        )

    def test_generate_dl_fail(self):
        with patch.object(wb, 'IntentionDetector', side_effect=lambda llm: self.mock_intention_detector), \
             patch.object(wb, 'SopGenerator', side_effect=lambda llm, ctx: self.mock_sop_generator), \
             patch.object(wb, 'ResourceRetriever', side_effect=lambda: self.mock_resource_retriever), \
             patch.object(wb, 'DLGenerator', side_effect=lambda llm, ctx: self.mock_dl_generator), \
             patch.object(wb, 'Reflector', side_effect=lambda: self.mock_reflector), \
             patch.object(wb, 'DLTransformer', side_effect=lambda llm, ctx: self.mock_transformer):
            builder = WorkflowBuilder(self.llm, self.context_manager)

        check_errors = ["error1", "error2"]
        def check_format_occur_errors(param):
            self.mock_reflector.errors = check_errors
        self.mock_reflector.check_format.side_effect = check_format_occur_errors
        with self.assertRaises(JiuWenBaseException) as cm:
            generated_dl = builder._generate_and_reflect_dl(
                dl_operation=builder._dl_generator.generate,
                query=wb.GENERATE_DL_FROM_SOP_CONTENT + TRANSFORMED_SOP,
                resource=None
            )
        self.assertIn(str(StatusCode.NL2AGENT_WORKFLOW_DL_GENERATION_ERROR.code), str(cm.exception))
        self.assertIn("流程定义语言（DL）生成失败", str(cm.exception))
        self.assertEqual(self.mock_dl_generator.generate.call_count, 3)
        self.mock_dl_generator.generate.assert_called_with(
            query=wb.GENERATE_DL_FROM_SOP_CONTENT + TRANSFORMED_SOP, resource=None
        )
        self.assertEqual(len(builder._dl_reflector.errors), 0)
        self.assertEqual(len(builder._dl_generator.reflect_prompts), 6)
        self.assertIn(GENERATED_DL, builder._dl_generator.reflect_prompts)
        self.assertIn(wb.MODIFY_DL_CONTENT + ";\n".join(check_errors), builder._dl_generator.reflect_prompts)

    def test_generate_dl_success(self):
        with patch.object(wb, 'IntentionDetector', side_effect=lambda llm: self.mock_intention_detector), \
             patch.object(wb, 'SopGenerator', side_effect=lambda llm, ctx: self.mock_sop_generator), \
             patch.object(wb, 'ResourceRetriever', side_effect=lambda: self.mock_resource_retriever), \
             patch.object(wb, 'DLGenerator', side_effect=lambda llm, ctx: self.mock_dl_generator), \
             patch.object(wb, 'Reflector', side_effect=lambda: self.mock_reflector), \
             patch.object(wb, 'DLTransformer', side_effect=lambda llm, ctx: self.mock_transformer):
            builder = WorkflowBuilder(self.llm, self.context_manager)

        generated_dl = builder._generate_and_reflect_dl(
            dl_operation=builder._dl_generator.refine,
            query="query",
            resource=None,
            exist_dl=GENERATED_DL,
            exist_mermaid=MERMAID
        )
        self.assertEqual(self.mock_dl_generator.refine.call_count, 1)
        self.mock_dl_generator.refine.assert_called_with(
            query="query", resource=None, exist_dl=GENERATED_DL, exist_mermaid=MERMAID
        )

    def test_refine_dl_fail(self):
        with patch.object(wb, 'IntentionDetector', side_effect=lambda llm: self.mock_intention_detector), \
             patch.object(wb, 'SopGenerator', side_effect=lambda llm, ctx: self.mock_sop_generator), \
             patch.object(wb, 'ResourceRetriever', side_effect=lambda: self.mock_resource_retriever), \
             patch.object(wb, 'DLGenerator', side_effect=lambda llm, ctx: self.mock_dl_generator), \
             patch.object(wb, 'Reflector', side_effect=lambda: self.mock_reflector), \
             patch.object(wb, 'DLTransformer', side_effect=lambda llm, ctx: self.mock_transformer):
            builder = WorkflowBuilder(self.llm, self.context_manager)

        check_errors = ["error1", "error2"]
        def check_format_occur_errors(param):
            self.mock_reflector.errors = check_errors
        self.mock_reflector.check_format.side_effect = check_format_occur_errors
        with self.assertRaises(JiuWenBaseException) as cm:
            generated_dl = builder._generate_and_reflect_dl(
                dl_operation=builder._dl_generator.refine,
                query="query",
                resource=None,
                exist_dl=GENERATED_DL,
                exist_mermaid=MERMAID
            )
        self.assertIn(str(StatusCode.NL2AGENT_WORKFLOW_DL_GENERATION_ERROR.code), str(cm.exception))
        self.assertIn("流程定义语言（DL）生成失败", str(cm.exception))
        self.assertEqual(self.mock_dl_generator.refine.call_count, 3)
        self.mock_dl_generator.refine.assert_called_with(
            query="query", resource=None, exist_dl=GENERATED_DL, exist_mermaid=MERMAID
        )
        self.assertEqual(len(builder._dl_reflector.errors), 0)
        self.assertEqual(len(builder._dl_generator.reflect_prompts), 6)
        self.assertIn(REFINED_DL, builder._dl_generator.reflect_prompts)
        self.assertIn(wb.MODIFY_DL_CONTENT + ";\n".join(check_errors), builder._dl_generator.reflect_prompts)


if __name__ == "__main__":
    unittest.main()