from typing import Union, Any, Optional

from jiuwen.core.agent.agent import Agent
from jiuwen.core.runtime.agent_group_manager import AgentGroupProvider, AgentGroupMgr
from jiuwen.core.runtime.agent_manager import AgentProvider, AgentMgr
from jiuwen.core.runtime.resource_manager import ResourceMgr
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.runtime.workflow import WorkflowRuntime
from jiuwen.core.runtime.wrapper import TaskRuntime
from jiuwen.core.utils.tool.base import Tool
from jiuwen.core.workflow.base import Workflow
from jiuwen.runner.agent_group import AgentGroup
from jiuwen.runner.message_queue import LocalMessageQueue, MessageQueue


class Runner:
    """
    Runner接口
    """

    def __init__(self, resource_manager: ResourceMgr, runner_id: str = ""):
        self._runner_id = runner_id
        self._resource_manager = resource_manager
        self._message_queue = LocalMessageQueue()
        self._agent_group_mgr: AgentGroupMgr = AgentGroupMgr()
        self._agent_mgr: AgentMgr = AgentMgr()

    def start(self) -> bool:
        pass

    def message_queue(self):
        return self._message_queue

    def add_agent_group(self, agent_group_id: str, agent_group: Union[AgentGroup, AgentGroupProvider]):
        # self._agent_mgr.add_agent_group(agent_group_id, agent_group)
        # if isinstance(agent_group, AgentGroup):
        #     topic = agent_group.get_topic()
        #     agent_group.set_subscription(self._message_queue.subscribe(topic))
        # else:
        #     agent_group_instance = self._agent_mgr.get_agent_group(agent_group_id)
        #     topic = agent_group_instance.get_topic()
        #     agent_group_instance.set_subscription(self._message_queue.subscribe(topic))
        pass

    def remove_agent_group(self, agent_group_id: str) -> Union[AgentGroup, AgentGroupProvider]:
        pass

    def add_agent(self, agent_id, agent: Union[Agent, AgentProvider]):
        pass

    def remove_agent(self, agent_id) -> Optional[Agent, AgentProvider]:
        pass

    async def run_workflow(self, workflow: Union[str, Workflow], inputs: Any,
                           *, runtime: Union[Runtime, WorkflowRuntime] = None):
        self._check_is_agent_workflow(runtime, workflow)
        workflow_runtime = self._create_workflow_runtime(runtime)
        pass

    async def run_workflow_streaming(self, workflow: Union[str, Workflow], inputs: Any,
                                     *, runtime: Union[Runtime, WorkflowRuntime] = None):
        self._check_is_agent_workflow(runtime, workflow)
        workflow_runtime = self._create_workflow_runtime(runtime)
        pass

    async def run_agent(self, agent: Union[str, Agent], inputs: Any):
        # agent_instance, agent_runtime = self._create_agent_runtime(agent, session_id)
        pass

    async def run_agent_streaming(self, agent: Union[str, Agent], inputs: Any):
        # agent_instance, agent_runtime = self._create_agent_runtime(agent, session_id)
        pass

    async def run_agent_group(self, agent_group: Union[str, AgentGroup], inputs: Any):
        pass

    async def run_agent_group_streaming(self, agent_group: Union[str, AgentGroup], inputs: Any):
        pass

    async def run_tool(self, tool: Union[str, Tool], inputs, *, runtime: Runtime = None):
        self._check_is_agent_tool(runtime, tool)
        # 调用
        pass

    async def run_tool_streaming(self, tool: Union[str, Tool], inputs, *, runtime: Runtime = None):
        self._check_is_agent_tool(runtime, tool)
        pass

    def _check_is_agent_tool(self, runtime, tool):
        if not self._is_called_by_agent(runtime):
            return
        pass

    def _check_is_agent_workflow(self, runtime, workflow):
        if not self._is_called_by_agent(runtime):
            return
        pass

    def _is_called_by_agent(self, runtime: Runtime) -> bool:
        return runtime and isinstance(runtime, TaskRuntime)

    def _create_workflow_runtime(self, runtime):
        # workflow的runtime的转换
        if not runtime:
            workflow_runtime = WorkflowRuntime()
        elif isinstance(runtime, TaskRuntime):
            workflow_runtime = runtime.create_workflow_runtime()
        else:
            workflow_runtime = runtime
        return workflow_runtime

    def _create_agent_runtime(self, session_id, agent: Union[str, Agent]):
        # if isinstance(agent, str):
        #     agent_with_runtime = self._agent_mgr.get_agent(agent)
        #     runtime = agent_with_runtime.runtime.pre_run(session_id=session_id)
        #     return agent_with_runtime.agent, runtime
        #
        # agent_runtime = AgentRuntime(agent.config())
        # return agent, agent_runtime.pre_run(session_id=session_id)
        pass
