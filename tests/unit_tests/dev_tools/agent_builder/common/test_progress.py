#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import unittest
from datetime import datetime
from unittest.mock import Mock, patch

from openjiuwen.dev_tools.agent_builder.common.progress import (
    ProgressStep, BuildProgress, ProgressReporter, ProgressManager, progress_stage
)
from openjiuwen.dev_tools.agent_builder.common.enums import ProgressStage, ProgressStatus, AgentType


class TestProgressStep(unittest.TestCase):
    """Test ProgressStep dataclass."""

    def test_creation_and_to_dict(self):
        """Test ProgressStep creation and serialization."""
        step = ProgressStep(
            stage=ProgressStage.INITIALIZING,
            status=ProgressStatus.RUNNING,
            message="Test message",
            details={"key": "value"}
        )
        step.duration = 1.5

        result = step.to_dict()

        self.assertEqual(result["stage"], "initializing")
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["message"], "Test message")
        self.assertEqual(result["details"], {"key": "value"})
        self.assertEqual(result["duration"], 1.5)
        self.assertIsNone(result["error"])


class TestBuildProgress(unittest.TestCase):
    """Test BuildProgress dataclass."""

    def test_creation_and_to_dict(self):
        """Test BuildProgress creation and serialization."""
        start_time = datetime(2025, 1, 1, 10, 0, 0)
        progress = BuildProgress(
            session_id="test-123",
            agent_type=AgentType.LLM_AGENT,
            current_stage=ProgressStage.GENERATING,
            current_status=ProgressStatus.RUNNING,
            current_message="Generating...",
            overall_progress=50.0,
            start_time=start_time,
            last_update_time=start_time
        )

        result = progress.to_dict()

        self.assertEqual(result["session_id"], "test-123")
        self.assertEqual(result["agent_type"], "llm_agent")
        self.assertEqual(result["current_stage"], "generating")
        self.assertEqual(result["current_status"], "running")
        self.assertEqual(result["overall_progress"], 50.0)
        self.assertEqual(result["start_time"], "2025-01-01 10:00:00")


class TestProgressReporter(unittest.TestCase):
    """Test ProgressReporter class."""

    def setUp(self):
        self.session_id = "test-session-123"
        self.agent_type = AgentType.LLM_AGENT

    def test_reporter_creation(self):
        """Test ProgressReporter initialization."""
        reporter = ProgressReporter(self.session_id, self.agent_type)

        self.assertEqual(reporter.session_id, self.session_id)
        self.assertEqual(reporter.agent_type, self.agent_type)
        self.assertEqual(reporter.progress.current_stage, ProgressStage.INITIALIZING)
        self.assertEqual(reporter.progress.current_status, ProgressStatus.PENDING)

    def test_callback_management(self):
        """Test callback add/remove."""
        reporter = ProgressReporter(self.session_id, self.agent_type)
        callback = Mock()

        reporter.add_callback(callback)
        self.assertIn(callback, reporter.callbacks)

        reporter.remove_callback(callback)
        self.assertNotIn(callback, reporter.callbacks)

    def test_stage_lifecycle(self):
        """Test start, update, complete, fail stage lifecycle."""
        reporter = ProgressReporter(self.session_id, self.agent_type)

        # Start stage
        reporter.start_stage(ProgressStage.RESOURCE_RETRIEVING, "Retrieving...")
        self.assertEqual(reporter.progress.current_stage, ProgressStage.RESOURCE_RETRIEVING)
        self.assertEqual(reporter.progress.current_status, ProgressStatus.RUNNING)
        self.assertEqual(len(reporter.progress.steps), 1)

        # Update stage
        reporter.update_stage(message="Still processing...", progress=50.0)
        self.assertEqual(reporter.progress.current_message, "Still processing...")
        self.assertEqual(reporter.progress.overall_progress, 50.0)

        # Complete stage
        reporter.complete_stage("Done", details={"count": 5})
        step = reporter.progress.steps[-1]
        self.assertEqual(step.status, ProgressStatus.SUCCESS)
        self.assertEqual(step.details, {"count": 5})
        self.assertIsNotNone(step.duration)

    def test_fail_stage(self):
        """Test fail_stage sets error correctly."""
        reporter = ProgressReporter(self.session_id, self.agent_type)
        reporter.start_stage(ProgressStage.CONVERTING, "Converting...")
        reporter.fail_stage("Conversion failed", details={"error_code": 500})

        step = reporter.progress.steps[-1]
        self.assertEqual(step.status, ProgressStatus.FAILED)
        self.assertEqual(step.error, "Conversion failed")
        self.assertEqual(reporter.progress.error, "Conversion failed")

    def test_callback_notification(self):
        """Test callback is notified and errors are handled."""
        reporter = ProgressReporter(self.session_id, self.agent_type)
        callback = Mock(side_effect=Exception("Callback error"))
        reporter.add_callback(callback)

        # Should not raise
        reporter.start_stage(ProgressStage.CLARIFYING, "Clarifying...")

    @patch('openjiuwen.dev_tools.agent_builder.common.progress.logger')
    def test_complete(self, mock_logger):
        """Test complete method."""
        reporter = ProgressReporter(self.session_id, self.agent_type)
        reporter.start_stage(ProgressStage.RESOURCE_RETRIEVING, "Retrieving...")
        reporter.complete("Done!")

        self.assertEqual(reporter.progress.current_status, ProgressStatus.SUCCESS)
        self.assertEqual(reporter.progress.overall_progress, 100.0)


class TestProgressManager(unittest.TestCase):
    """Test ProgressManager class."""

    def setUp(self):
        self.manager = ProgressManager()

    def test_reporter_lifecycle(self):
        """Test create, get, remove reporter."""
        # Create
        reporter = self.manager.create_reporter("session-1", "llm_agent")
        self.assertIsNotNone(reporter)
        self.assertEqual(reporter.session_id, "session-1")

        # Get same instance
        reporter2 = self.manager.get_reporter("session-1")
        self.assertIs(reporter, reporter2)

        # Remove
        self.manager.remove_reporter("session-1")
        self.assertIsNone(self.manager.get_reporter("session-1"))

    def test_get_progress(self):
        """Test getting progress from manager."""
        self.manager.create_reporter("session-2", "workflow")
        progress = self.manager.get_progress("session-2")

        self.assertIsNotNone(progress)
        self.assertEqual(progress.session_id, "session-2")

        # Nonexistent returns None
        self.assertIsNone(self.manager.get_progress("nonexistent"))


class TestProgressStageDecorator(unittest.TestCase):
    """Test progress_stage decorator."""

    def setUp(self):
        self.mock_reporter = Mock(spec=ProgressReporter)
        self.mock_reporter.start_stage = Mock()
        self.mock_reporter.complete_stage = Mock()
        self.mock_reporter.fail_stage = Mock()

    def test_decorator_basic(self):
        """Test decorator with and without reporter."""

        class TestBuilderWithReporter:
            progress_reporter = self.mock_reporter

            @progress_stage(
                stage=ProgressStage.GENERATING,
                start_message="Starting generation",
                complete_message="Generation complete",
                fail_message="Generation failed"
            )
            def build_method(self):
                return "result"

        class TestBuilderWithoutReporter:
            @progress_stage(
                stage=ProgressStage.VALIDATING,
                start_message="Starting validation",
                complete_message="Validation complete",
                fail_message="Validation failed"
            )
            def build_method(self):
                return "result"

        # With reporter
        builder1 = TestBuilderWithReporter()
        result = builder1.build_method()
        self.assertEqual(result, "result")
        self.mock_reporter.start_stage.assert_called_once()
        self.mock_reporter.complete_stage.assert_called_once()

        # Without reporter
        builder2 = TestBuilderWithoutReporter()
        result = builder2.build_method()
        self.assertEqual(result, "result")

    @patch('openjiuwen.dev_tools.agent_builder.common.progress.logger')
    def test_decorator_error_handling(self, mock_logger):
        """Test decorator handles errors and calls fail_stage."""

        class TestBuilder:
            progress_reporter = self.mock_reporter

            @progress_stage(
                stage=ProgressStage.CONVERTING,
                start_message="Starting conversion",
                complete_message="Conversion complete",
                fail_message="Conversion failed"
            )
            def build_method(self):
                raise ValueError("Conversion error")

        builder = TestBuilder()

        with self.assertRaises(ValueError):
            builder.build_method()

        self.mock_reporter.fail_stage.assert_called_once()

    def test_decorator_with_detail_builder(self):
        """Test decorator with custom detail builder."""

        class TestBuilder:
            progress_reporter = self.mock_reporter

            @progress_stage(
                stage=ProgressStage.RESOURCE_RETRIEVING,
                start_message="Retrieving",
                complete_message="Retrieved",
                fail_message="Retrieval failed",
                detail_builder=lambda self, result: {"result_count": len(result) if result else 0}
            )
            def build_method(self):
                return ["item1", "item2"]

        builder = TestBuilder()
        builder.build_method()

        self.mock_reporter.complete_stage.assert_called_once()
        call_kwargs = self.mock_reporter.complete_stage.call_args[1]
        self.assertEqual(call_kwargs["details"], {"result_count": 2})


class TestProgressReporterStageTransition(unittest.TestCase):
    """Test stage transition behavior in ProgressReporter."""

    def test_multiple_stages_and_custom_progress(self):
        """Test tracking multiple stages and custom progress."""
        reporter = ProgressReporter("test-session", "llm_agent")

        reporter.start_stage(ProgressStage.INITIALIZING, "Initializing...")
        reporter.complete_stage("Initialized")

        reporter.start_stage(ProgressStage.RESOURCE_RETRIEVING, "Retrieving...", progress=25.0)
        self.assertEqual(reporter.progress.overall_progress, 25.0)
        reporter.complete_stage("Retrieved")

        self.assertEqual(len(reporter.progress.steps), 2)
        self.assertEqual(reporter.progress.steps[0].stage, ProgressStage.INITIALIZING)
        self.assertEqual(reporter.progress.steps[1].stage, ProgressStage.RESOURCE_RETRIEVING)


if __name__ == '__main__':
    unittest.main()
