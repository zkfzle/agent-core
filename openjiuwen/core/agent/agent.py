#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Iterator, Dict, List, Union, AsyncIterator, TYPE_CHECKING

from openjiuwen.agent.common.schema import WorkflowSchema
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.logging import logger
from openjiuwen.core.context_engine.config import ContextEngineConfig
from openjiuwen.core.context_engine.engine import ContextEngine
from openjiuwen.core.runtime.agent import StaticAgentRuntime
from openjiuwen.core.runtime.resource_manager import ResourceMgr
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.runtime.workflow_manager import generate_workflow_key
from openjiuwen.core.runtime.config import Config
from openjiuwen.core.runtime.wrapper import WrappedRuntime, StaticWrappedRuntime, TaskRuntime
from openjiuwen.core.stream.base import OutputSchema
from openjiuwen.core.utils.config.user_config import UserConfig
from openjiuwen.core.utils.tool.base import Tool
from openjiuwen.core.utils.tool.function.function import LocalFunction
from openjiuwen.core.utils.tool.service_api.restful_api import RestfulApi
from openjiuwen.core.workflow.base import Workflow

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
    The top-level abstract class and the common base class for all Agents.
    Subclasses must implement:
        - invoke : synchronous one-time call
        - stream : streaming call
    """

    def __init__(self, config: Config) -> None:
        # 所有核心属性统一在基类初始化
        self._config = config
        self._runtime = AgentRuntime(config=config)
        self._context_engine = self._create_context_engine()
        self._message_handler_class = None

    def config(self) -> Config:
        """获取 Agent 配置"""
        return self._config

    @property
    def context_engine(self) -> ContextEngine:
        """获取 Context Engine - 统一公共接口"""
        return self._context_engine

    def set_message_handler(self, message_handler_class):
        """设置 MessageHandler 类 - 子类在构造函数中调用此方法注入具体实现
        
        Args:
            message_handler_class: MessageHandler 的类（不是实例）
        """
        self._message_handler_class = message_handler_class

    def get_message_handler(self):
        """获取当前的 MessageHandler 类"""
        return self._message_handler_class

    @abstractmethod
    async def invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        pass

    @abstractmethod
    async def stream(self, inputs: Dict, runtime: Runtime = None) -> Iterator[Any]:
        pass

    def _create_context_engine(self) -> ContextEngine:
        """创建 ContextEngine - 内部方法，在基类初始化时调用"""
        context_config = ContextEngineConfig(
            conversation_history_length=self._config.get_agent_config().constrain.reserved_max_chat_rounds * 2
        )
        return ContextEngine(
            agent_id=self._config.get_agent_config().id,
            config=context_config,
            model=None
        )

    def _create_controller(self, runtime: Runtime) -> "Controller":
        """创建 Controller 实例"""
        from openjiuwen.core.agent.controller.controller import Controller
        
        if self._message_handler_class is None:
            return None

        # 实例化 MessageHandler
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
        """解包 OutputSchema 结果 - 子类可直接调用的公共方法"""
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
        """基于 Controller 的同步调用 - 子类可直接调用的公共方法"""
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
        """基于 Controller 的流式调用 - 子类可直接调用的公共方法"""
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
        """适配runner的接口，待所有agent的使用都完全适配runner后，接口改为controller_stream替换旧接口"""
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
        self._runtime.add_workflows(
            [(generate_workflow_key(workflow.config().metadata.id, workflow.config().metadata.version), workflow) for
             workflow in
             workflows])
        for workflow in workflows:
            self._config.get_agent_config().workflows.append(WorkflowSchema(id=workflow.config().metadata.id,
                                                         name=workflow.config().metadata.name,
                                                         version=workflow.config().metadata.version,
                                                         description=workflow.config().metadata.description))

    def bind_tools(self, tools: List[Tool]):
        self._runtime.add_tools(
            [(tool.name, tool) for tool in tools if (isinstance(tool, RestfulApi) or isinstance(tool, LocalFunction))])
        for tool in tools:
            self._config.get_agent_config().tools.append(tool.name)

    def get_llm_calls(self) -> Dict:
        raise NotImplementedError("")

    def copy(self) -> "Agent":
        raise NotImplementedError("")


# ===== 新架构：BaseAgent 和 ControllerAgent =====

class BaseAgent(ABC):
    """基础 Agent - 极简接口定义（新架构）
    """

    def __init__(self, agent_config):
        """初始化 Agent
        
        Args:
            agent_config: Agent 配置
        """
        from openjiuwen.core.runtime.config import Config
        
        # 1. 创建 Config 包装器（向后兼容）
        self._config_wrapper = Config()
        self._config_wrapper.set_agent_config(agent_config)
        self._agent_config = agent_config
        self._config = self._config_wrapper  # 统一接口
        
        # 2. 创建 Runtime
        self._runtime = AgentRuntime(config=self._config)
        
        # 3. 创建 ContextEngine
        self._context_engine = self._create_context_engine()
        
        # 4. 统一持有 tools 和 workflows（消除子类重复）
        self._tools: List[Tool] = []
        self._workflows: List[Workflow] = []

    def config(self) -> Config:
        """获取 Config 包装器 - 向后兼容的方法接口
        
        Returns:
            Config 实例（包含 get_agent_config() 方法）
        """
        return self._config_wrapper
    
    @property
    def tools(self) -> List[Tool]:
        """获取工具列表 - 子类只读访问"""
        return self._tools
    
    @property
    def workflows(self) -> List[Workflow]:
        """获取工作流列表 - 子类只读访问"""
        return self._workflows
    
    @property
    def context_engine(self) -> ContextEngine:
        """获取 Context Engine - 统一公共接口"""
        return self._context_engine
    
    def _create_context_engine(self) -> ContextEngine:
        """创建 ContextEngine - 内部方法，在基类初始化时调用"""
        context_config = ContextEngineConfig(
            conversation_history_length=self._agent_config.constrain.reserved_max_chat_rounds * 2
            if hasattr(self._agent_config, 'constrain') and hasattr(self._agent_config.constrain, 'reserved_max_chat_rounds')
            else 20  # 默认值
        )
        return ContextEngine(
            agent_id=self._agent_config.id,
            config=context_config,
            model=None
        )

    @abstractmethod
    async def invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        """同步调用入口 - 抽象方法
        
        子类必须实现此方法
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 必须实现 invoke() 方法"
        )

    @abstractmethod
    async def stream(self, inputs: Dict, runtime: Runtime = None) -> AsyncIterator[Any]:
        """流式调用入口 - 抽象方法
        
        子类必须实现此方法
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 必须实现 stream() 方法"
        )

    # ===== 动态配置接口（方案A：向后兼容） =====
    
    def add_prompt(self, prompt_template: List[Dict]) -> None:
        """添加 Prompt 模板
        
        Args:
            prompt_template: Prompt 模板列表，每个元素是 dict，如 {"role": "system", "content": "..."}
        
        注意：
        - 此方法仅更新配置，不影响已创建的 runtime
        - 子类如需同步 runtime，应重写此方法
        """
        # 检查配置是否有 prompt_template 字段
        if hasattr(self._agent_config, 'prompt_template'):
            # 追加模式：保留原有 prompt，添加新 prompt
            self._agent_config.prompt_template.extend(prompt_template)
        else:
            logger.warning(
                f"{self._agent_config.__class__.__name__} 没有 prompt_template 字段，"
                "add_prompt 操作被忽略"
            )

    def add_tools(self, tools: List[Tool]) -> None:
        """添加工具（同时更新 config、runtime、self._tools）
        """
        from openjiuwen.agent.common.schema import PluginSchema
        
        for tool in tools:
            # 1. 添加工具名到 config.tools
            if tool.name not in self._agent_config.tools:
                self._agent_config.tools.append(tool.name)
            
            # 2. 生成 PluginSchema（如果配置支持）
            if hasattr(self._agent_config, 'plugins'):
                # 检查是否已存在
                existing_names = {p.name for p in self._agent_config.plugins}
                if tool.name not in existing_names:
                    plugin_schema = self._tool_to_plugin_schema(tool)
                    self._agent_config.plugins.append(plugin_schema)
            
            # 3. 添加到 self._tools（避免重复）
            existing_tool_names = {t.name for t in self._tools}
            if tool.name not in existing_tool_names:
                self._tools.append(tool)
            
            # 4. 同步到 runtime（自动注册）
            self._runtime.add_tools([(tool.name, tool)])

    def add_workflows(self, workflows: List[Workflow]) -> None:
        """添加工作流（同时更新 config、runtime、self._workflows）
        
        Args:
            workflows: 工作流实例列表
        """
        from openjiuwen.agent.common.schema import WorkflowSchema
        
        for workflow in workflows:
            # 生成 WorkflowSchema
            workflow_config = workflow.config()
            workflow_key = f"{workflow_config.metadata.id}_{workflow_config.metadata.version}"
            
            # 检查是否已存在
            existing_keys = {
                f"{w.id}_{w.version}" for w in self._agent_config.workflows
            }
            if workflow_key not in existing_keys:
                # 1. 更新 config.workflows
                workflow_schema = WorkflowSchema(
                    id=workflow_config.metadata.id,
                    name=workflow_config.metadata.name,
                    version=workflow_config.metadata.version,
                    description=workflow_config.metadata.description,
                    inputs={}
                )
                self._agent_config.workflows.append(workflow_schema)
                
                # 2. 添加到 self._workflows
                self._workflows.append(workflow)
                
                # 3. 同步到 runtime（自动注册）
                self._runtime.add_workflows([(workflow_key, workflow)])

    def add_plugins(self, plugins: List) -> None:
        """添加插件 Schema
        
        Args:
            plugins: PluginSchema 列表
        
        注意：
        - 此方法仅更新配置中的 plugins 字段
        - 子类如需同步 runtime，应重写此方法
        """
        if hasattr(self._agent_config, 'plugins'):
            # 检查重复
            existing_names = {p.name for p in self._agent_config.plugins}
            for plugin in plugins:
                if plugin.name not in existing_names:
                    self._agent_config.plugins.append(plugin)
                    existing_names.add(plugin.name)
        else:
            logger.warning(
                f"{self._agent_config.__class__.__name__} 没有 plugins 字段，"
                "add_plugins 操作被忽略"
            )

    def _tool_to_plugin_schema(self, tool: Tool):
        """将 Tool 实例转换为 PluginSchema
        
        这是内部方法，用于自动生成 plugin schema
        """
        from openjiuwen.agent.common.schema import PluginSchema
        
        # 从 tool.params 生成 inputs
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
        
        return PluginSchema(
            id=tool.name,
            name=tool.name,
            description=tool.description if hasattr(tool, 'description') else "",
            inputs=inputs
        )


class ControllerAgent(BaseAgent):
    """持有 Controller 的 Agent（新架构）
    """

    def __init__(self, agent_config, controller=None):
        """初始化 ControllerAgent
        
        Args:
            agent_config: Agent 配置
            controller: 可选的 Controller（如果不提供，子类应该在 invoke/stream 时创建）
        """
        super().__init__(agent_config)
        self.controller = controller
        
        # 如果传入了 controller，确保 controller 有 agent 引用
        if self.controller:
            self.controller.agent = self

    async def invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        """同步调用 - 完全委托给 controller
        
        Args:
            inputs: 输入数据
            runtime: Runtime 实例
        
        Returns:
            执行结果
        """
        if not self.controller:
            raise RuntimeError(
                f"{self.__class__.__name__} 没有 controller，"
                "子类应该在调用前创建 controller"
            )
        
        # 完全委托给 controller
        return await self.controller.invoke(inputs, runtime)

    async def stream(self, inputs: Dict, runtime: Runtime = None) -> AsyncIterator[Any]:
        """流式调用 - 完全委托给 controller
        
        Args:
            inputs: 输入数据
            runtime: Runtime 实例
        
        Yields:
            流式输出
        """
        if not self.controller:
            raise RuntimeError(
                f"{self.__class__.__name__} 没有 controller，"
                "子类应该在调用前创建 controller"
            )
        
        # 完全委托给 controller
        async def stream_process():
            await self.controller.invoke(inputs, runtime)

        task = asyncio.create_task(stream_process())
        async for result in runtime.stream_iterator():
            yield result
        await task
