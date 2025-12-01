#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import asyncio
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Union

from openjiuwen.agent.common.schema import WorkflowSchema, PluginSchema
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.logging import logger
from openjiuwen.core.context_engine.config import ContextEngineConfig
from openjiuwen.core.context_engine.engine import ContextEngine
from openjiuwen.core.runtime.agent import StaticAgentRuntime
from openjiuwen.core.runtime.resources_manager.resource_manager import ResourceMgr
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.runtime.resources_manager.workflow_manager import generate_workflow_key
from openjiuwen.core.runtime.wrapper import (
    StaticWrappedRuntime,
    TaskRuntime,
    WrappedRuntime
)
from openjiuwen.core.stream.base import OutputSchema
from openjiuwen.core.common.security.user_config import UserConfig
from openjiuwen.core.utils.tool.base import Tool
from openjiuwen.core.utils.tool.function.function import LocalFunction
from openjiuwen.core.utils.tool.service_api.restful_api import RestfulApi
from openjiuwen.core.workflow.base import Workflow, WorkflowOutput
from openjiuwen.core.runtime.config import Config

if TYPE_CHECKING:
    from openjiuwen.core.agent.controller.controller import Controller


class AgentRuntime(WrappedRuntime, StaticWrappedRuntime):
    """
    deprecated
    """

    def __init__(self, config: Config = None, resource_mgr: ResourceMgr = None):
        inner = StaticAgentRuntime(config, resource_mgr=resource_mgr)
        super().__init__(inner)
        self._runtime = inner

    async def write_stream(self, data: Union[dict, OutputSchema]):
        return await self.write_custom_stream(data)

    async def pre_run(self, **kwargs) -> Runtime:
        session_id = kwargs.get("session_id")
        if session_id is None:
            session_id = kwargs.get("trace_id")
        inputs = kwargs.get("inputs")
        inner = await self._runtime.create_agent_runtime(session_id, inputs)
        return TaskRuntime(inner=inner)

    async def release(self, session_id: str):
        await self._runtime.checkpointer().release(session_id)


class Agent(ABC):
    """
    DEPRECATED: This class is deprecated and will be removed in a future version.
    
    Please use BaseAgent or ControllerAgent instead:
    - BaseAgent: Minimal interface for new agent implementations
    - ControllerAgent: Agent with built-in controller support
    
    Legacy documentation:
    The top-level abstract class and the common base class for all Agents.
    Subclasses must implement:
        - invoke : synchronous one-time call
        - stream : streaming call
    """

    def __init__(self, config: Config) -> None:
        # Emit deprecation warning
        warnings.warn(
            f"{self.__class__.__name__} inherits from deprecated Agent class. "
            "Please migrate to BaseAgent or ControllerAgent. "
            "Agent class will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # All core attributes initialized uniformly in base class
        self._config = config
        self._runtime = AgentRuntime(config=config)
        self._context_engine = self._create_context_engine()
        self._message_handler_class = None

    def config(self) -> Config:
        """Get Agent configuration"""
        return self._config

    @property
    def context_engine(self) -> ContextEngine:
        """Get Context Engine - Unified public interface"""
        return self._context_engine

    def set_message_handler(self, message_handler_class):
        """Set MessageHandler class - Subclasses call this method in constructor to inject concrete implementation
        
        Args:
            message_handler_class: MessageHandler class (not instance)
        """
        self._message_handler_class = message_handler_class

    def get_message_handler(self):
        """Get current MessageHandler class"""
        return self._message_handler_class

    @abstractmethod
    async def invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        pass

    @abstractmethod
    async def stream(self, inputs: Dict, runtime: Runtime = None) -> Iterator[Any]:
        pass

    async def forward(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        """Agent's forward method, invoke the agent with inputs."""
        return await self.invoke(inputs, runtime)

    def _create_context_engine(self) -> ContextEngine:
        """Create ContextEngine - Internal method, called during base class initialization"""
        agent_config = self._config.get_agent_config()
        max_rounds = agent_config.constrain.reserved_max_chat_rounds
        context_config = ContextEngineConfig(
            conversation_history_length=max_rounds * 2
        )
        return ContextEngine(
            agent_id=self._config.get_agent_config().id,
            config=context_config,
            model=None
        )

    def _create_controller(self, runtime: Runtime) -> "Controller":
        """Create Controller instance"""
        from openjiuwen.core.agent.controller.controller import Controller

        if self._message_handler_class is None:
            return None

        # Instantiate MessageHandler
        message_handler = self._message_handler_class(
            self._config.get_agent_config(),
            self._context_engine,
            runtime
        )

        return Controller(
            self._config.get_agent_config(),
            self._context_engine,
            runtime,
            message_handler
        )

    @staticmethod
    def unwrap_result(result):
        """Unwrap OutputSchema result - Public method that subclasses can call directly"""
        if isinstance(result, list):
            if not result:
                return result
            if isinstance(result[0], OutputSchema):
                if len(result) == 1 and result[0].type == "workflow_final":
                    return result[0].payload
                return result
            return result

        if isinstance(result, OutputSchema):
            payload = result.payload
            if isinstance(payload, dict):
                if 'output' in payload and isinstance(payload['output'], str):
                    payload['output'] = payload['output'].strip()
            return payload

        return result

    async def controller_invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        """Controller-based synchronous invocation - Public method that subclasses can call directly"""
        session_id = inputs.pop("conversation_id", "default_session")

        if runtime is None:
            agent_runtime = await self._runtime.pre_run(session_id=session_id)
        else:
            agent_runtime = runtime

        controller = None
        try:
            controller = self._create_controller(agent_runtime)
            await controller.start()
            result = await controller.process_inputs(inputs)

            logger.info("Controller completed, closing stream")

            if runtime is None:
                await agent_runtime.post_run()

            if result is not None:
                return self.unwrap_result(result)

            return {"output": "Message processed by scheduler", "result_type": "answer"}
        except Exception as e:
            if UserConfig.is_sensitive():
                logger.info(f"{self.__class__.__name__} invoke error.")
            else:
                logger.error(f"{self.__class__.__name__} invoke error: {e}")
            raise
        finally:
            if controller:
                await controller.stop()

    async def controller_stream(self, inputs: Dict, runtime: Runtime = None) -> AsyncIterator[Any]:
        """Controller-based streaming invocation - Public method that subclasses can call directly"""
        session_id = inputs.pop("conversation_id", "default_session")

        if runtime is None:
            agent_runtime = await self._runtime.pre_run(session_id=session_id)
        else:
            agent_runtime = runtime

        async def stream_process():
            controller = None
            try:
                controller = self._create_controller(agent_runtime)
                await controller.start()
                await controller.process_inputs(inputs)
                logger.info("Controller completed, workflow done")
            except Exception as e:
                if UserConfig.is_sensitive():
                    logger.info(f"{self.__class__.__name__} stream error.")
                else:
                    logger.error(f"{self.__class__.__name__} stream error: {e}")
                raise JiuWenBaseException(StatusCode.AGENT_SUB_TASK_TYPE_ERROR.code,
                                          f"{self.__class__.__name__} stream error.")
            finally:
                if controller:
                    await controller.stop()
                if runtime is None:
                    await agent_runtime.post_run()

        task = asyncio.create_task(stream_process())
        async for result in agent_runtime.stream_iterator():
            yield result

        try:
            await task
        except Exception as e:
            logger.error(f"{self.__class__.__name__} stream error.")
            if UserConfig.is_sensitive():
                raise JiuWenBaseException(StatusCode.AGENT_SUB_TASK_TYPE_ERROR.code,
                                          f"{self.__class__.__name__} stream error.")
            else:
                raise JiuWenBaseException(StatusCode.AGENT_SUB_TASK_TYPE_ERROR.code,
                                          f"{self.__class__.__name__} stream error.") from e

    async def runner_controller_stream(self, inputs: Dict, runtime: Runtime):
        """Interface adapted for runner, will be replaced with controller_stream after all agents fully adapt to runner"""
        controller = None
        try:
            controller = self._create_controller(runtime)
            await controller.start()
            await controller.process_inputs(inputs)
            logger.info("Controller completed, workflow done")
        except Exception as e:
            if UserConfig.is_sensitive():
                logger.info(f"{self.__class__.__name__} stream error.")
            else:
                logger.error(f"{self.__class__.__name__} stream error: {e}")
            raise JiuWenBaseException(StatusCode.AGENT_SUB_TASK_TYPE_ERROR.code,
                                      f"{self.__class__.__name__} stream error.")
        finally:
            if controller:
                await controller.stop()

    def bind_workflows(self, workflows: List[Workflow]):
        """Bind workflows to Agent
        
        Args:
            workflows: List of workflow instances
        """
        workflow_items = [
            (generate_workflow_key(
                workflow.config().metadata.id,
                workflow.config().metadata.version
            ), workflow)
            for workflow in workflows
        ]
        self._runtime.add_workflows(workflow_items)

        for workflow in workflows:
            metadata = workflow.config().metadata
            workflow_schema = WorkflowSchema(
                id=metadata.id,
                name=metadata.name,
                version=metadata.version,
                description=metadata.description
            )
            self._config.get_agent_config().workflows.append(workflow_schema)

    def bind_tools(self, tools: List[Tool]):
        """Bind tools to Agent
        
        Args:
            tools: List of tool instances
        """
        # Filter only supported tool types
        tool_items = [
            (tool.name, tool)
            for tool in tools
            if isinstance(tool, (RestfulApi, LocalFunction))
        ]
        self._runtime.add_tools(tool_items)

        for tool in tools:
            self._config.get_agent_config().tools.append(tool.name)

    def get_llm_calls(self) -> Dict:
        raise NotImplementedError("")

    def copy(self) -> "Agent":
        raise NotImplementedError("")


class BaseAgent(ABC):
    """Base Agent - Minimal interface definition (new architecture)
    """

    def __init__(self, agent_config):
        """Initialize Agent
        
        Args:
            agent_config: Agent configuration
        """
        # 1. Create Config wrapper (backward compatible)
        self._config_wrapper = Config()
        self._config_wrapper.set_agent_config(agent_config)
        self._agent_config = agent_config
        self._config = self._config_wrapper  # Unified interface

        # 2. Create Runtime
        self._runtime = AgentRuntime(config=self._config)

        # 3. Create ContextEngine
        self._context_engine = self._create_context_engine()

        # 4. Uniformly hold tools and workflows (eliminate subclass duplication)
        self._tools: List[Tool] = []
        self._workflows: List[Workflow] = []

    def config(self) -> Config:
        """Get Config wrapper - Backward compatible method interface
        
        Returns:
            Config instance (contains get_agent_config() method)
        """
        return self._config_wrapper

    @property
    def tools(self) -> List[Tool]:
        """Get tools list - Read-only access for subclasses"""
        return self._tools

    @property
    def workflows(self) -> List[Workflow]:
        """Get workflows list - Read-only access for subclasses"""
        return self._workflows

    @property
    def context_engine(self) -> ContextEngine:
        """Get Context Engine - Unified public interface"""
        return self._context_engine

    def _create_context_engine(self) -> ContextEngine:
        """Create ContextEngine - Internal method, called during base class initialization"""
        # Get max conversation rounds configuration
        if (hasattr(self._agent_config, 'constrain') and
                hasattr(self._agent_config.constrain, 'reserved_max_chat_rounds')):
            max_rounds = self._agent_config.constrain.reserved_max_chat_rounds
        else:
            max_rounds = 10  # Default value

        context_config = ContextEngineConfig(
            conversation_history_length=max_rounds * 2
        )
        return ContextEngine(
            agent_id=self._agent_config.id,
            config=context_config,
            model=None
        )

    @abstractmethod
    async def invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        """Synchronous invocation entry point - Abstract method
        
        Subclasses must implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement invoke() method"
        )

    @abstractmethod
    async def stream(self, inputs: Dict, runtime: Runtime = None) -> AsyncIterator[Any]:
        """Streaming invocation entry point - Abstract method
        
        Subclasses must implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement stream() method"
        )

    # ===== Dynamic configuration interface (Plan A: Backward compatible) =====

    def add_prompt(self, prompt_template: List[Dict]) -> None:
        """Add Prompt template
        
        Args:
            prompt_template: Prompt template list, format like
                [{"role": "system", "content": "..."}]
        
        Note:
        - This method only updates configuration, does not affect already created runtime
        - Subclasses should override this method if they need to sync runtime
        """
        # Check if configuration has prompt_template field
        if hasattr(self._agent_config, 'prompt_template'):
            # Append mode: Keep original prompt, add new prompt
            self._agent_config.prompt_template.extend(prompt_template)
        else:
            config_class_name = self._agent_config.__class__.__name__
            logger.warning(
                f"{config_class_name} has no prompt_template field, "
                "add_prompt operation ignored"
            )

    def add_tools(self, tools: List[Tool]) -> None:
        """Add tools (update config, runtime, and self._tools simultaneously)
        
        Args:
            tools: List of tool instances
        """

        for tool in tools:
            # 1. Add tool name to config.tools
            if tool.name not in self._agent_config.tools:
                self._agent_config.tools.append(tool.name)

            # 2. Generate PluginSchema (if configuration supports)
            if hasattr(self._agent_config, 'plugins'):
                # Check if already exists
                existing_plugin_names = {
                    p.name for p in self._agent_config.plugins
                }
                if tool.name not in existing_plugin_names:
                    plugin_schema = self._tool_to_plugin_schema(tool)
                    self._agent_config.plugins.append(plugin_schema)

            # 3. Add to self._tools (avoid duplication)
            existing_tool_names = {t.name for t in self._tools}
            if tool.name not in existing_tool_names:
                self._tools.append(tool)

            # 4. Sync to runtime (auto register)
            self._runtime.add_tools([(tool.name, tool)])

    def add_workflows(self, workflows: List[Workflow]) -> None:
        """Add workflows (update config, runtime, and self._workflows simultaneously)
        
        Args:
            workflows: List of workflow instances
        """
        logger.info(f"BaseAgent.add_workflows called with {len(workflows)} workflows")

        for workflow in workflows:
            # Generate WorkflowSchema
            workflow_config = workflow.config()
            workflow_key = generate_workflow_key(
                workflow_config.metadata.id,
                workflow_config.metadata.version
            )

            # Check if already exists
            existing_keys = {
                generate_workflow_key(w.id, w.version)
                for w in self._agent_config.workflows
            }
            logger.info(
                f"Workflow {workflow_key}: existing_keys={existing_keys}, exists={workflow_key in existing_keys}")

            # Even if schema exists, still need to add workflow instance
            if workflow_key not in existing_keys:
                # 1. Update config.workflows
                workflow_schema = WorkflowSchema(
                    id=workflow_config.metadata.id,
                    name=workflow_config.metadata.name,
                    version=workflow_config.metadata.version,
                    description=workflow_config.metadata.description,
                    inputs={}
                )
                self._agent_config.workflows.append(workflow_schema)

            # 2. Add to self._workflows (if not exists)
            if workflow not in self._workflows:
                self._workflows.append(workflow)

            # 3. Sync to runtime (auto register)
            self._runtime.add_workflows([(workflow_key, workflow)])

            # 4. Also add to global Runner.resource_mgr (for cross-runtime access)
            try:
                from openjiuwen.core.runner.runner import resource_mgr
                logger.info(f"Adding workflow {workflow_key} to global resource_mgr")
                resource_mgr.workflow().add_workflow(workflow_key, workflow)
                logger.info(f"Successfully added workflow {workflow_key} to global resource_mgr")
            except Exception as e:
                logger.error(f"Failed to add workflow to global resource_mgr: {e}")

    def bind_workflows(self, workflows: List[Workflow]) -> None:
        """Bind workflows - Backward compatible alias method
        
        Args:
            workflows: List of workflow instances
        """
        self.add_workflows(workflows)

    def add_plugins(self, plugins: List) -> None:
        """Add plugin Schema
        
        Args:
            plugins: PluginSchema list
        
        Note:
        - This method only updates plugins field in configuration
        - Subclasses should override this method if they need to sync runtime
        """
        if hasattr(self._agent_config, 'plugins'):
            # Check duplication
            existing_names = {p.name for p in self._agent_config.plugins}
            for plugin in plugins:
                if plugin.name not in existing_names:
                    self._agent_config.plugins.append(plugin)
                    existing_names.add(plugin.name)
        else:
            config_class_name = self._agent_config.__class__.__name__
            logger.warning(
                f"{config_class_name} has no plugins field, "
                "add_plugins operation ignored"
            )

    def _tool_to_plugin_schema(self, tool: Tool):
        """Convert Tool instance to PluginSchema
        
        This is an internal method for automatically generating plugin schema
        
        Args:
            tool: Tool instance
            
        Returns:
            PluginSchema: Plugin schema object
        """
        # Generate inputs from tool.params
        inputs = {
            "type": "object",
            "properties": {},
            "required": []
        }

        if hasattr(tool, 'params') and tool.params:
            for param in tool.params:
                prop = {
                    "type": param.type,
                    "description": param.description
                }
                inputs["properties"][param.name] = prop
                if param.required:
                    inputs["required"].append(param.name)

        tool_description = ""
        if hasattr(tool, 'description'):
            tool_description = tool.description

        return PluginSchema(
            id=tool.name,
            name=tool.name,
            description=tool_description,
            inputs=inputs
        )

    async def clear_session(self, session_id: str = "default_session"):
        await self._runtime.release(session_id)


class ControllerAgent(BaseAgent):
    """Agent that holds Controller (new architecture)
    """

    def __init__(self, agent_config, controller=None):
        """Initialize ControllerAgent
        
        Args:
            agent_config: Agent configuration
            controller: Optional Controller instance (will be auto-configured)
            
        Note:
            If controller is provided, it will be automatically configured with
            config, context_engine and runtime from this agent via setup_from_agent()
            
        Usage:
            # Simplest way - controller auto-configured:
            controller = WorkflowController()  # No parameters needed
            agent = ControllerAgent(config=config, controller=controller)
            
            # Alternative - set controller after agent creation:
            agent = ControllerAgent(config=config)
            agent.controller = WorkflowController()  # Will be auto-configured
        """
        super().__init__(agent_config)
        self.controller = controller
        
        # Auto-configure controller if provided
        if self.controller is not None:
            self._setup_controller()
    
    def _setup_controller(self):
        """Setup controller with agent's config, context_engine and runtime"""
        if hasattr(self.controller, 'setup_from_agent'):
            self.controller.setup_from_agent(self)
    
    @property
    def controller(self):
        """Get controller"""
        return self._controller
    
    @controller.setter
    def controller(self, value):
        """Set controller and auto-configure it"""
        self._controller = value
        # Auto-configure when setting controller
        # Only if agent is already initialized (has _context_engine)
        if value is not None and hasattr(self, '_context_engine'):
            self._setup_controller()

    async def invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        """Synchronous invocation - Fully delegate to controller
        
        Args:
            inputs: Input data
            runtime: Runtime instance (if None, auto create)
        
        Returns:
            Execution result
        """
        if not self.controller:
            raise RuntimeError(
                f"{self.__class__.__name__} has no controller, "
                "subclass should create controller before invocation"
            )

        # If runtime not provided, create one
        session_id = inputs.get("conversation_id", "default_session")
        if runtime is None:
            agent_runtime = await self._runtime.pre_run(session_id=session_id)
        else:
            agent_runtime = runtime

        try:
            # Fully delegate to controller
            result = await self.controller.invoke(inputs, agent_runtime)
            if runtime is None:
                await agent_runtime.post_run()

            return result
        except Exception as e:
            await agent_runtime.post_run()
            raise

    async def stream(self, inputs: Dict, runtime: Runtime = None) -> AsyncIterator[Any]:
        """Streaming invocation - Fully delegate to controller
        
        Args:
            inputs: Input data
            runtime: Runtime instance (if None, auto create)
        
        Yields:
            Streaming output
        
        Note:
            当传入外部 runtime 时，数据会写入该 runtime，但不从其 stream_iterator
            读取（避免嵌套读取导致死锁）。外部调用方负责从 runtime 读取流式数据。
        """
        if not self.controller:
            raise RuntimeError(
                f"{self.__class__.__name__} has no controller, "
                "subclass should create controller before invocation"
            )

        # If runtime not provided, create one
        session_id = inputs.get("conversation_id", "default_session")
        if runtime is None:
            agent_runtime = await self._runtime.pre_run(session_id=session_id)
            need_cleanup = True
            own_stream = True  # 自己拥有 stream 的生命周期
        else:
            agent_runtime = runtime
            need_cleanup = False
            own_stream = False  # 外部拥有 stream 的生命周期

        # 用于存储最终结果，供 send_to_agent 获取
        final_result_holder = {"result": None}

        # Fully delegate to controller
        async def stream_process():
            try:
                res = await self.controller.invoke(inputs, agent_runtime)
                final_result_holder["result"] = res
                # 中断情况：list 包含 __interaction__ 等 OutputSchema
                # 流式数据（包括 workflow_final）已由 controller 层写入 runtime
                if isinstance(res, list):
                    for item in res:
                        await agent_runtime.write_stream(item)
            finally:
                if need_cleanup:
                    await agent_runtime.post_run()

        task = asyncio.create_task(stream_process())
        
        if own_stream:
            # 只有自己拥有 stream 时才从 stream_iterator 读取
            # 如果传入了外部 runtime，外部调用方负责读取
            async for result in agent_runtime.stream_iterator():
                yield result
        
        await task
        
        # 当 own_stream = False 时，yield 最终结果给 send_to_agent
        # 这样 send_to_agent 可以获取到 agent 的实际返回值
        if not own_stream and final_result_holder["result"] is not None:
            res = final_result_holder["result"]
            if isinstance(res, list):
                # 中断情况：返回 list（包含 __interaction__）
                for item in res:
                    yield item
            else:
                # 正常完成：yield dict 或其他结果
                yield res
