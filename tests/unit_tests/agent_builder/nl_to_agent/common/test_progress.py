#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import unittest
from datetime import datetime
from unittest.mock import Mock, patch

from openjiuwen.agent_builder.nl_to_agent.common.progress import (
    ProgressStep,
    BuildProgress,
    ProgressReporter,
    ProgressManager,
    progress_stage,
    resource_retrieving_stage,
)
from openjiuwen.agent_builder.nl_to_agent.common.enums import ProgressStage, ProgressStatus


class TestProgressStep(unittest.TestCase):
    """Test ProgressStep dataclass."""

    def test_progress_step_creation(self):
        """Test basic ProgressStep creation."""
        step = ProgressStep(
            stage=ProgressStage.INITIALIZING,
            status=ProgressStatus.RUNNING,
            message="Test message",
            details={"key": "value"}
        )

        self.assertEqual(step.stage, ProgressStage.INITIALIZING)
        self.assertEqual(step.status, ProgressStatus.RUNNING)
        self.assertEqual(step.message, "Test message")
        self.assertEqual(step.details, {"key": "value"})
        self.assertIsNotNone(step.timestamp)
        self.assertIsNone(step.duration)
        self.assertIsNone(step.error)

    def test_progress_step_to_dict(self):
        """Test ProgressStep to_dict method."""
        step = ProgressStep(
            stage=ProgressStage.RESOURCE_RETRIEVING,
            status=ProgressStatus.SUCCESS,
            message="Resource retrieved",
            details={"count": 5}
        )
        step.duration = 1.5

        result = step.to_dict()

        self.assertEqual(result["stage"], "resource_retrieving")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["message"], "Resource retrieved")
        self.assertEqual(result["details"], {"count": 5})
        self.assertEqual(result["duration"], 1.5)
        self.assertIsNone(result["error"])


class TestBuildProgress(unittest.TestCase):
    """Test BuildProgress dataclass."""

    def test_build_progress_creation(self):
        """Test basic BuildProgress creation."""
        progress = BuildProgress(
            session_id="test-123",
            agent_type="llm_agent",
            current_stage=ProgressStage.INITIALIZING,
            current_status=ProgressStatus.PENDING,
            current_message="Starting..."
        )

        self.assertEqual(progress.session_id, "test-123")
        self.assertEqual(progress.agent_type, "llm_agent")
        self.assertEqual(progress.current_stage, ProgressStage.INITIALIZING)
        self.assertEqual(progress.current_status, ProgressStatus.PENDING)
        self.assertEqual(progress.current_message, "Starting...")
        self.assertEqual(progress.steps, [])
        self.assertEqual(progress.overall_progress, 0.0)
        self.assertIsNotNone(progress.start_time)
        self.assertIsNotNone(progress.last_update_time)
        self.assertIsNone(progress.error)

    def test_build_progress_to_dict(self):
        """Test BuildProgress to_dict method."""
        start_time = datetime(2025, 1, 1, 10, 0, 0)
        progress = BuildProgress(
            session_id="test-456",
            agent_type="workflow",
            current_stage=ProgressStage.GENERATING,
            current_status=ProgressStatus.RUNNING,
            current_message="Generating...",
            overall_progress=50.0,
            start_time=start_time,
            last_update_time=start_time
        )

        result = progress.to_dict()

        self.assertEqual(result["session_id"], "test-456")
        self.assertEqual(result["agent_type"], "workflow")
        self.assertEqual(result["current_stage"], "generating")
        self.assertEqual(result["current_status"], "running")
        self.assertEqual(result["current_message"], "Generating...")
        self.assertEqual(result["overall_progress"], 50.0)
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["start_time"], "2025-01-01 10:00:00")
        self.assertEqual(result["last_update_time"], "2025-01-01 10:00:00")


class TestProgressReporter(unittest.TestCase):
    """Test ProgressReporter class."""

    def setUp(self):
        self.session_id = "test-session-123"
        self.agent_type = "llm_agent"

    def test_reporter_creation(self):
        """Test ProgressReporter initialization."""
        reporter = ProgressReporter(self.session_id, self.agent_type)

        self.assertEqual(reporter.session_id, self.session_id)
        self.assertEqual(reporter.agent_type, self.agent_type)
        self.assertEqual(reporter.progress.current_stage, ProgressStage.INITIALIZING)
        self.assertEqual(reporter.progress.current_status, ProgressStatus.PENDING)
        self.assertEqual(reporter.progress.current_message, "Initializing...")
        self.assertEqual(reporter.progress.steps, [])
        self.assertEqual(reporter._callbacks, [])

    def test_add_and_remove_callback(self):
        """Test callback management."""
        reporter = ProgressReporter(self.session_id, self.agent_type)
        callback = Mock()

        reporter.add_callback(callback)
        self.assertIn(callback, reporter._callbacks)

        reporter.remove_callback(callback)
        self.assertNotIn(callback, reporter._callbacks)

    def test_get_progress(self):
        """Test get_progress returns correct progress."""
        reporter = ProgressReporter(self.session_id, self.agent_type)

        progress = reporter.get_progress()

        self.assertIs(progress, reporter.progress)
        self.assertEqual(progress.session_id, self.session_id)

    @patch('openjiuwen.agent_builder.nl_to_agent.common.progress.logger')
    def test_complete(self, mock_logger):
        """Test complete method."""
        reporter = ProgressReporter(self.session_id, self.agent_type)
        reporter.start_stage(ProgressStage.RESOURCE_RETRIEVING, "Retrieving...")
        reporter.complete("Done!")

        self.assertEqual(reporter.progress.current_stage, ProgressStage.COMPLETED)
        self.assertEqual(reporter.progress.current_status, ProgressStatus.SUCCESS)
        self.assertEqual(reporter.progress.current_message, "Done!")
        self.assertEqual(reporter.progress.overall_progress, 100.0)
        mock_logger.info.assert_called()

    @patch('openjiuwen.agent_builder.nl_to_agent.common.progress.logger')
    def test_start_stage(self, mock_logger):
        """Test start_stage method."""
        reporter = ProgressReporter(self.session_id, self.agent_type)

        reporter.start_stage(
            ProgressStage.RESOURCE_RETRIEVING,
            "Retrieving resources...",
            details={"tool_count": 5}
        )

        self.assertEqual(reporter.progress.current_stage, ProgressStage.RESOURCE_RETRIEVING)
        self.assertEqual(reporter.progress.current_status, ProgressStatus.RUNNING)
        self.assertEqual(reporter.progress.current_message, "Retrieving resources...")
        self.assertEqual(len(reporter.progress.steps), 1)
        self.assertEqual(reporter.progress.steps[0].details, {"tool_count": 5})
        mock_logger.info.assert_called()

    def test_update_stage(self):
        """Test update_stage method."""
        reporter = ProgressReporter(self.session_id, self.agent_type)
        reporter.start_stage(ProgressStage.GENERATING, "Generating...")

        reporter.update_stage(
            message="Still generating...",
            details={"progress": 50},
            progress=60.0
        )

        self.assertEqual(reporter.progress.current_message, "Still generating...")
        self.assertEqual(reporter.progress.steps[-1].details, {"progress": 50})
        self.assertEqual(reporter.progress.overall_progress, 60.0)

    def test_update_stage_partial(self):
        """Test update_stage with partial updates."""
        reporter = ProgressReporter(self.session_id, self.agent_type)
        reporter.start_stage(ProgressStage.VALIDATING, "Validating...")
        original_progress = reporter.progress.overall_progress

        reporter.update_stage(message="Updated message")

        self.assertEqual(reporter.progress.current_message, "Updated message")
        self.assertEqual(reporter.progress.overall_progress, original_progress)

    @patch('openjiuwen.agent_builder.nl_to_agent.common.progress.logger')
    def test_complete_stage(self, mock_logger):
        """Test complete_stage method."""
        reporter = ProgressReporter(self.session_id, self.agent_type)
        reporter.start_stage(ProgressStage.VALIDATING, "Validating...")
        reporter.complete_stage("Validation passed", details={"errors": 0})

        step = reporter.progress.steps[-1]
        self.assertEqual(step.status, ProgressStatus.SUCCESS)
        self.assertEqual(step.message, "Validation passed")
        self.assertEqual(step.details, {"errors": 0})

    @patch('openjiuwen.agent_builder.nl_to_agent.common.progress.logger')
    def test_fail_stage(self, mock_logger):
        """Test fail_stage method."""
        reporter = ProgressReporter(self.session_id, self.agent_type)
        reporter.start_stage(ProgressStage.CONVERTING, "Converting...")
        reporter.fail_stage("Conversion failed", "Error occurred", details={"error_code": 500})

        step = reporter.progress.steps[-1]
        self.assertEqual(step.status, ProgressStatus.FAILED)
        self.assertEqual(step.message, "Error occurred")
        self.assertEqual(step.error, "Conversion failed")
        self.assertEqual(reporter.progress.error, "Conversion failed")

    def test_notify_callbacks(self):
        """Test callback notification."""
        reporter = ProgressReporter(self.session_id, self.agent_type)
        callback = Mock()
        reporter.add_callback(callback)

        reporter.start_stage(ProgressStage.CLARIFYING, "Clarifying...")

        callback.assert_called_once()
        self.assertIsInstance(callback.call_args[0][0], BuildProgress)

    def test_notify_callback_error(self):
        """Test callback error handling."""
        reporter = ProgressReporter(self.session_id, self.agent_type)
        callback = Mock(side_effect=Exception("Callback error"))
        reporter.add_callback(callback)

        # Should not raise exception
        reporter.start_stage(ProgressStage.CLARIFYING, "Clarifying...")

    def test_calculate_progress_llm_agent(self):
        """Test progress calculation for llm_agent."""
        reporter = ProgressReporter(self.session_id, "llm_agent")

        self.assertEqual(reporter._calculate_progress(ProgressStage.INITIALIZING), 0.0)
        self.assertEqual(reporter._calculate_progress(ProgressStage.RESOURCE_RETRIEVING), 20.0)
        self.assertEqual(reporter._calculate_progress(ProgressStage.CLARIFYING), 40.0)
        self.assertEqual(reporter._calculate_progress(ProgressStage.GENERATING), 70.0)
        self.assertEqual(reporter._calculate_progress(ProgressStage.CONVERTING), 90.0)
        self.assertEqual(reporter._calculate_progress(ProgressStage.COMPLETED), 100.0)

    def test_calculate_progress_workflow(self):
        """Test progress calculation for workflow."""
        reporter = ProgressReporter(self.session_id, "workflow")

        self.assertEqual(reporter._calculate_progress(ProgressStage.INITIALIZING), 0.0)
        self.assertEqual(reporter._calculate_progress(ProgressStage.RESOURCE_RETRIEVING), 10.0)
        self.assertEqual(reporter._calculate_progress(ProgressStage.CLARIFYING), 20.0)
        self.assertEqual(reporter._calculate_progress(ProgressStage.GENERATING), 40.0)
        self.assertEqual(reporter._calculate_progress(ProgressStage.VALIDATING), 60.0)
        self.assertEqual(reporter._calculate_progress(ProgressStage.CONVERTING), 80.0)
        self.assertEqual(reporter._calculate_progress(ProgressStage.COMPLETED), 100.0)

    def test_calculate_progress_unknown_stage(self):
        """Test progress calculation for unknown stage."""
        reporter = ProgressReporter(self.session_id, "llm_agent")

        self.assertEqual(reporter._calculate_progress(ProgressStage.ERROR), 0.0)


class TestProgressManager(unittest.TestCase):
    """Test ProgressManager class."""

    def setUp(self):
        self.manager = ProgressManager()

    def test_create_reporter(self):
        """Test creating a new reporter."""
        reporter = self.manager.create_reporter("session-1", "llm_agent")

        self.assertIsNotNone(reporter)
        self.assertEqual(reporter.session_id, "session-1")
        self.assertEqual(reporter.agent_type, "llm_agent")

    def test_get_existing_reporter(self):
        """Test getting existing reporter returns same instance."""
        reporter1 = self.manager.create_reporter("session-2", "workflow")
        reporter2 = self.manager.get_reporter("session-2")

        self.assertIs(reporter1, reporter2)

    def test_get_nonexistent_reporter(self):
        """Test getting nonexistent reporter returns None."""
        reporter = self.manager.get_reporter("nonexistent")

        self.assertIsNone(reporter)

    def test_remove_reporter(self):
        """Test removing a reporter."""
        self.manager.create_reporter("session-3", "llm_agent")
        self.manager.remove_reporter("session-3")

        reporter = self.manager.get_reporter("session-3")
        self.assertIsNone(reporter)

    def test_remove_nonexistent_reporter(self):
        """Test removing nonexistent reporter does not raise error."""
        # Should not raise
        self.manager.remove_reporter("nonexistent")

    def test_get_progress(self):
        """Test getting progress from manager."""
        self.manager.create_reporter("session-4", "llm_agent")
        progress = self.manager.get_progress("session-4")

        self.assertIsNotNone(progress)
        self.assertEqual(progress.session_id, "session-4")

    def test_get_progress_nonexistent(self):
        """Test getting progress from nonexistent session."""
        progress = self.manager.get_progress("nonexistent")

        self.assertIsNone(progress)


class TestProgressStageDecorator(unittest.TestCase):
    """Test progress_stage decorator."""

    def setUp(self):
        self.mock_reporter = Mock(spec=ProgressReporter)
        self.mock_reporter.start_stage = Mock()
        self.mock_reporter.complete_stage = Mock()
        self.mock_reporter.fail_stage = Mock()

    def test_decorator_with_reporter(self):
        """Test decorator with progress reporter."""

        class TestBuilder:
            progress_reporter = self.mock_reporter

            @progress_stage(
                stage=ProgressStage.GENERATING,
                start_message="Starting generation",
                complete_message="Generation complete",
                fail_message="Generation failed"
            )
            def build_method(self):
                return "result"

        builder = TestBuilder()
        result = builder.build_method()

        self.assertEqual(result, "result")
        self.mock_reporter.start_stage.assert_called_once()
        self.mock_reporter.complete_stage.assert_called_once()

    def test_decorator_without_reporter(self):
        """Test decorator without progress reporter."""

        class TestBuilder:
            @progress_stage(
                stage=ProgressStage.VALIDATING,
                start_message="Starting validation",
                complete_message="Validation complete",
                fail_message="Validation failed"
            )
            def build_method(self):
                return "result"

        builder = TestBuilder()
        result = builder.build_method()

        self.assertEqual(result, "result")

    @patch('openjiuwen.agent_builder.nl_to_agent.common.progress.logger')
    def test_decorator_with_error(self, mock_logger):
        """Test decorator handles errors correctly."""

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

    def test_decorator_raise_on_error_false(self):
        """Test decorator with raise_on_error=False."""

        class TestBuilder:
            progress_reporter = self.mock_reporter

            @progress_stage(
                stage=ProgressStage.GENERATING,
                start_message="Generating",
                complete_message="Generated",
                fail_message="Generation failed",
                raise_on_error=False
            )
            def build_method(self):
                raise RuntimeError("Error")

        builder = TestBuilder()
        result = builder.build_method()

        self.assertIsNone(result)


class TestResourceRetrievingStageDecorator(unittest.TestCase):
    """Test resource_retrieving_stage decorator."""

    def setUp(self):
        self.mock_reporter = Mock(spec=ProgressReporter)
        self.mock_reporter.start_stage = Mock()
        self.mock_reporter.complete_stage = Mock()
        self.mock_reporter.fail_stage = Mock()

    def test_decorator_creates_correct_stage(self):
        """Test decorator uses correct stage and messages."""

        class TestBuilder:
            progress_reporter = self.mock_reporter

            @resource_retrieving_stage()
            def retrieve_method(self):
                return {"tools": []}

        builder = TestBuilder()
        builder.retrieve_method()

        self.mock_reporter.start_stage.assert_called_once()
        call_args = self.mock_reporter.start_stage.call_args
        self.assertEqual(call_args[0][0], ProgressStage.RESOURCE_RETRIEVING)
        self.assertEqual(call_args[0][1], "Retrieving resources...")

        self.mock_reporter.complete_stage.assert_called_once()
        call_args = self.mock_reporter.complete_stage.call_args
        self.assertEqual(call_args[0][0], "Successfully retrieved resources.")

    def test_decorator_with_custom_detail_builder(self):
        """Test decorator with custom detail builder."""

        class TestBuilder:
            progress_reporter = self.mock_reporter

            @resource_retrieving_stage(
                detail_builder=lambda self, result: {"retrieved": result}
            )
            def retrieve_method(self):
                return {"tools": ["tool1"]}

        builder = TestBuilder()
        builder.retrieve_method()

        self.mock_reporter.complete_stage.assert_called_once()
        call_kwargs = self.mock_reporter.complete_stage.call_args[1]
        self.assertEqual(call_kwargs["details"], {"retrieved": {"tools": ["tool1"]}})


class TestProgressReporterStageTransition(unittest.TestCase):
    """Test stage transition behavior in ProgressReporter."""

    def test_stage_duration_recorded(self):
        """Test that stage duration is recorded on completion."""
        reporter = ProgressReporter("test-session", "llm_agent")
        reporter.start_stage(ProgressStage.RESOURCE_RETRIEVING, "Retrieving...")
        reporter.complete_stage("Done")

        step = reporter.progress.steps[-1]
        self.assertEqual(step.status, ProgressStatus.SUCCESS)
        self.assertIsNotNone(step.duration)

    def test_multiple_stages(self):
        """Test tracking multiple stages."""
        reporter = ProgressReporter("test-session", "llm_agent")

        reporter.start_stage(ProgressStage.INITIALIZING, "Initializing...")
        reporter.complete_stage("Initialized")

        reporter.start_stage(ProgressStage.RESOURCE_RETRIEVING, "Retrieving...")
        reporter.complete_stage("Retrieved")

        self.assertEqual(len(reporter.progress.steps), 2)
        self.assertEqual(reporter.progress.steps[0].stage, ProgressStage.INITIALIZING)
        self.assertEqual(reporter.progress.steps[1].stage, ProgressStage.RESOURCE_RETRIEVING)

    def test_custom_progress_in_start_stage(self):
        """Test setting custom progress in start_stage."""
        reporter = ProgressReporter("test-session", "llm_agent")

        reporter.start_stage(
            ProgressStage.GENERATING,
            "Generating...",
            progress=75.0
        )

        self.assertEqual(reporter.progress.overall_progress, 75.0)


if __name__ == '__main__':
    unittest.main()
