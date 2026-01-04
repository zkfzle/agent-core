"""Single Agent Base Class Definition

Main Classes:
 - Ability: Ability type definition
 - AbilityKit: Agent ability manager
 - BaseAgent: Single agent base class

Created on: 2025-11-25
Author: huenrui1@huawei.com
"""
from __future__ import annotations

from abc import abstractmethod, ABC
from typing import List, Any, AsyncIterator, Union, Optional, Tuple, Dict, TYPE_CHECKING

from openjiuwen.core.foundation.llm import ToolMessage
from openjiuwen.core.foundation.tool import ToolCall, ToolInfo
from openjiuwen.core.foundation.tool.base import ToolCard
from openjiuwen.core.protocols.mcp import McpServerConfig
from openjiuwen.core.single_agent.schema.agent_card import AgentCard

if TYPE_CHECKING:
    from openjiuwen.core.session import Session
    from openjiuwen.core.workflow import WorkflowCard

# Ability type definition (use string for WorkflowCard to avoid circular import at runtime)
Ability = Union[ToolCard, 'WorkflowCard', AgentCard, McpServerConfig]


class AbilityKit:
    """Agent能力管理器

    职责:
    - 存储Agent可用的能力Card（只存元数据，不存实例）
    - 提供能力的增删查接口
    - 将Card转换为ToolInfo供LLM使用
    - 执行能力调用（从ResourceManager获取实例）
    """

    def __init__(self):
        self._tools: Dict[str, ToolCard] = {}
        self._workflows: Dict[str, WorkflowCard] = {}
        self._agents: Dict[str, AgentCard] = {}
        self._mcp_servers: Dict[str, McpServerConfig] = {}

    def add(self, ability: Ability) -> None:
        """添加能力"""
        # TODO: 根据ability类型添加到对应的dict
        pass

    def remove(self, name: str) -> Optional[Ability]:
        """移除能力"""
        # TODO: 从各个dict中查找并移除
        pass

    def get(self, name: str) -> Optional[Ability]:
        """获取能力Card"""
        # TODO: 从各个dict中查找
        pass

    def list(self) -> List[Ability]:
        """列出所有能力Card"""
        # TODO: 从各个dict中收集所有Card
        pass

    def list_tool_info(
            self,
            names: Optional[List[str]] = None,
            mcp_server_name: Optional[str] = None
    ) -> List[ToolInfo]:
        """获取ToolInfo列表（供LLM使用）"""
        # TODO: 将Card转换为ToolInfo
        pass

    async def execute(
            self,
            tool_call: ToolCall,
            session: Session
    ) -> Tuple[Any, ToolMessage]:
        """执行能力调用

        从ResourceManager获取实例，执行并返回结果
        """
        # TODO: 
        # 1. 根据tool_call.name查找是哪种能力
        # 2. 从Runner().resource_mgr获取实例
        # 3. 执行并返回结果
        pass


class BaseAgent(ABC):
    """单Agent基类

    设计原则:
    - Card必需（定义Agent是什么）
    - Config可选（定义Agent怎么运行）
    - 所有配置方法支持链式调用

    Attributes:
        card: Agent名片（必需）
        _ability_kit: 能力管理器
    """

    def __init__(
            self,
            card: AgentCard,
    ):
        """初始化Agent

        Args:
            card: Agent名片（必需）
            config: Agent配置（可选，有默认值）
            context_engine: 上下文引擎（可选）
        """
        self.card = card
        self._ability_kit = AbilityKit()

    # ========== 配置接口 ==========
    @abstractmethod
    def configure(self, config) -> 'BaseAgent':
        """设置配置"""
        pass

    # ========== 能力管理接口 ==========

    def add_ability(self, ability: Union[Ability, List[Ability]]) -> 'BaseAgent':
        """添加能力

        Args:
            ability: 能力Card或列表（ToolCard/WorkflowCard/AgentCard/McpServerConfig）

        Returns:
            self（支持链式调用）
        """
        abilities = [ability] if not isinstance(ability, list) else ability
        for ab in abilities:
            self._ability_kit.add(ab)
        return self

    def remove_ability(self, name: Union[str, List[str]]) -> 'BaseAgent':
        """移除能力

        Args:
            name: 能力名称或列表

        Returns:
            self（支持链式调用）
        """
        names = [name] if isinstance(name, str) else name
        for n in names:
            self._ability_kit.remove(n)
        return self

    def get_ability(self, name: str) -> Optional[Ability]:
        """获取能力Card

        Args:
            name: 能力名称

        Returns:
            能力Card，如果不存在返回None
        """
        return self._ability_kit.get(name)

    def list_abilities(self) -> List[Ability]:
        """列出所有能力Card

        Returns:
            能力Card列表
        """
        return self._ability_kit.list()

    # ========== 查询接口 ==========
    def get_tool_info(self) -> ToolInfo:
        """将当前Agent转换为ToolInfo（作为子Agent使用）"""
        # TODO: 从self.card构造ToolInfo
        pass

    # ========== 执行接口 ==========

    async def _execute_ability(
            self,
            tool_calls: Union[ToolCall, List[ToolCall]],
            session: Session
    ) -> List[Tuple[Any, ToolMessage]]:
        """执行能力调用（支持并行）"""
        # TODO:
        # 1. 将单个tool_call转为列表
        # 2. 并行调用self._ability_kit.execute()
        # 3. 返回结果列表
        pass

    @abstractmethod
    async def invoke(
            self,
            inputs: Any,
            session: Optional[Session] = None,
    ) -> Any:
        """批执行（运行时可传入config覆盖）

        Args:
            inputs: Agent输入，支持以下格式：
                - dict: 必须包含 "user_input" 和 "session_id"
                   例如: {"user_input": "xxx", "session_id": "session_123"}
                - str: 直接作为user_input，需要单独传入session或通过其他方式获取session_id
            session: 会话对象（可选，如果不传会根据inputs中的session_id创建）

        Returns:
            Agent输出结果
        """
        ...

    @abstractmethod
    async def stream(
            self,
            inputs: Any,
            session: Optional[Session] = None,
            stream_modes: Optional[List[StreamMode]] = None
    ) -> AsyncIterator[Any]:
        """流式执行（运行时可传入config覆盖）

        Args:
            inputs: Agent输入，支持以下格式：
                - dict: 必须包含 "user_input" 和 "session_id"
                   例如: {"user_input": "xxx", "session_id": "session_123"}
                - str: 直接作为user_input，需要单独传入session或通过其他方式获取session_id
            session: 会话对象（可选，如果不传会根据inputs中的session_id创建）
            stream_modes: 流式输出模式（可选）

        Yields:
            Agent流输出结果
        """
        ...
