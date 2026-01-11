# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from requests import session

from openjiuwen.core.common import BaseCard
from openjiuwen.core.common.exception.status_code import StatusCode
from pydantic import BaseModel

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.foundation.prompt import PromptTemplate
from openjiuwen.core.foundation.tool import Tool, ToolInfo, ToolCard
from openjiuwen.core.multi_agent import BaseGroup, GroupCard
from openjiuwen.core.protocols.mcp import McpServerConfig
from openjiuwen.core.runner.drunner.remote_client.remote_agent import RemoteAgent
from openjiuwen.core.runner.resources_manager.base import (
    AgentGroupProvider,
    Error,
    Ok,
    Tag,
    GLOBAL,
    Result,
    TagUpdateStrategy,
    TagMatchStrategy,
    AgentProvider,
    WorkflowProvider,
    ModelProvider)
from openjiuwen.core.runner.resources_manager.resource_registry import ResourceRegistry

from typing import Optional, Union, Tuple

from openjiuwen.core.runner.resources_manager.tag_manager import TagMgr
from openjiuwen.core.runner.resources_manager.thread_safe_dict import ThreadSafeDict
from openjiuwen.core.session import Session
from openjiuwen.core.single_agent import BaseAgent, AgentCard
from openjiuwen.core.workflow.workflow import Workflow
from openjiuwen.core.workflow import WorkflowCard


class ResourceMgr:
    """
    Resource Manager for Model, Workflow, Prompt, Tool
    """

    def __init__(self, ) -> None:
        self._resource_registry = ResourceRegistry()
        self._tag_mgr = TagMgr()
        self._id_to_card: ThreadSafeDict[str, BaseCard] = {}

    async def add_agent_group(self,
                              card: GroupCard,
                              agent_group: AgentGroupProvider,
                              *,
                              tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                              tag_update_strategy: TagUpdateStrategy = TagUpdateStrategy.MERGE
                              ) -> Result[GroupCard, Exception]:
        """
        Add a single agent group to the resource manager.

        Args:
            card: The agent group's metadata card containing configuration and identification.
            agent_group: Callable provider that creates or returns the agent group instance.
            tag: Optional tag(s) for categorizing and filtering the agent group.
                 If None, no tags will be added or updated.
            tag_update_strategy: Strategy for updating tags when resource already exists.
                MERGE - Add new tags while keeping existing ones (default).
                REPLACE - Replace all existing tags with new tags.

        Returns:
            Result[GroupCard, Exception]: Result object containing the added group card or an exception.
        """
        try:
            if not self._resource_registry.is_id_unique(card.id):
                raise JiuWenBaseException(
                    StatusCode.SESSION_RESOURCE_REGISTRY_FAILED.code,
                    StatusCode.SESSION_RESOURCE_REGISTRY_FAILED.errmsg.format(
                        reason=f"When registering resource, id should be unique"
                    )
                )
            await self._resource_registry.agent_group().add_agent_group(card.id, agent_group)
            if tag is not None:
                self._tag_mgr.replace_resource_tags(card.id, tag, tag_update_strategy)
            self._id_to_card[card.id] = card
            return Ok(card)
        except Exception as e:
            return Error(e)

    async def remove_agent_group(self,
                                 *,
                                 id: Optional[Union[str, list[str]]] = None,
                                 tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                                 tag_match_strategy: TagMatchStrategy = TagMatchStrategy.ALL,
                                 skip_if_not_exists: bool = False,
                                 ) -> Result[Optional[GroupCard], Exception] | list[
        Result[Optional[GroupCard], Exception]]:
        """
        Remove agent group(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of agent groups to remove.
                Cannot be used together with tag parameter.
            tag: Single tag or list of tags; removes all agent groups with matching tags.
                Cannot be used together with id parameter.
            tag_match_strategy: Strategy for matching tags when using tag parameter.
                ALL - Resource must have all specified tags.
                ANY - Resource must have at least one of the specified tags.
            skip_if_not_exists: If True, silently skip non-existent resources.
                If False, raise ResourceNotFoundError for non-existent resources.

        Returns:
            Result[Optional[GroupCard], Exception] or list[Result[Optional[GroupCard], Exception]]:
                Result object(s) containing the removed group card(s) or exception.
        """
        if id and tag:
            raise JiuWenBaseException(
                StatusCode.SESSION_AGENT_GROUP_REMOVE_FAILED.code,
                StatusCode.SESSION_AGENT_GROUP_REMOVE_FAILED.errmsg.format(
                    reason=f"When removing agent group, id and tag parameter cannot be used together with tag parameter."
                )
            )
        ids_to_remove = []
        remove_results = []
        if isinstance(id, str):
            ids_to_remove = [id]
        elif isinstance(id, list):
            ids_to_remove = id
        elif id is None:
            ids_to_remove = self._tag_mgr.find_resources_by_tags(tag, tag_match_strategy=tag_match_strategy)
        for _id in ids_to_remove:
            try:
                if not self._tag_mgr.has_resource(_id) and not skip_if_not_exists:
                    raise JiuWenBaseException(
                        StatusCode.SESSION_AGENT_GROUP_REMOVE_FAILED.code,
                        StatusCode.SESSION_AGENT_GROUP_REMOVE_FAILED.errmsg.format(
                            reason=f"{_id} is not existent."
                        )
                    )
                self._tag_mgr.untag_resource(_id)
                card = self._id_to_card.pop(_id, None)
                if card is None and not skip_if_not_exists:
                    raise JiuWenBaseException(
                        StatusCode.SESSION_AGENT_GROUP_REMOVE_FAILED.code,
                        StatusCode.SESSION_AGENT_GROUP_REMOVE_FAILED.errmsg.format(
                            reason=f"{_id} card is not existent"
                        )
                    )
                await self._resource_registry.agent_group().remove_agent_group(agent_group_id=_id)
                remove_results.append(Ok(card))
            except Exception as e:
                remove_results.append(Error(e))

        return remove_results

    async def get_agent_group(self,
                              *,
                              id: str = None,
                              tag: Optional[Tag] = None,
                              tag_match_strategy: TagMatchStrategy = TagMatchStrategy.ALL,
                              session: Optional[Session] = None
                              ) -> Optional[BaseGroup]:
        """
        Get an agent group instance by ID or tag.

        Args:
            id: Unique identifier of the agent group. Either id or tag must be provided.
            tag: Optional tag for filtering when id is provided,
                 or main lookup criteria when id is not provided.
            tag_match_strategy: Strategy for matching tags when using tag parameter.
            session: Optional session context for the agent group.
                If provided, the agent group will be initialized with this session.

        Returns:
            BaseGroup instance if found, None otherwise.

        Raises:
            ValueError: When neither id nor tag is provided.
        """
        if id is None and tag is None:
            raise ValueError("Either id or tag must be provided.")
        if id:
            if tag is not None:
                matched_ids = self._tag_mgr.find_resources_by_tags(tag, tag_match_strategy=tag_match_strategy)
                if id not in matched_ids:
                    return None
            return self._resource_registry.agent_group().get_agent_group(agent_group_id=id, session=session)
        matched_ids = self._tag_mgr.find_resources_by_tags(tag, tag_match_strategy=tag_match_strategy)
        if not matched_ids:
            return None
        if len(matched_ids) > 1:
            raise ValueError("Got multiple agent group matched ids.")
        return self._resource_registry.agent_group().get_agent_group(
            agent_group_id=matched_ids[0],
            session=session
        )

    def add_agent(self,
                  card: AgentCard,
                  agent: AgentProvider | RemoteAgent,
                  *,
                  tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                  tag_update_strategy: TagUpdateStrategy = TagUpdateStrategy.MERGE
                  ) -> Result[AgentCard, Exception]:
        """
        Add a single agent to the resource manager.

        Args:
            card: The agent's metadata card containing configuration and identification.
            agent: Callable provider that creates or returns the agent instance.
            tag: Optional tag(s) for categorizing and filtering the agent.
            tag_update_strategy: Strategy for updating tags when agent already exists.

        Returns:
            Result[AgentCard, Exception]: Result object containing the added agent card or an exception.

        """
        try:
            if not self._resource_registry.is_id_unique(card.id):
                raise JiuWenBaseException(
                    StatusCode.SESSION_RESOURCE_REGISTRY_FAILED.code,
                    StatusCode.SESSION_RESOURCE_REGISTRY_FAILED.errmsg.format(
                        reason=f"When registering resource, id should be unique"
                    )
                )
            self._resource_registry.agent().add_agent(agent_id=card.id, agent=agent)
            self._tag_mgr.replace_resource_tags(card.id, tag, tag_update_strategy)
            return Ok(card)
        except Exception as e:
            return Error(e)

    def add_agents(self,
                   agents: list[Tuple[AgentCard, AgentProvider]],
                   *,
                   tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                   tag_update_strategy: TagUpdateStrategy = TagUpdateStrategy.MERGE
                   ) -> Result[AgentCard, Exception] | list[Result[AgentCard, Exception]]:
        """
        Add multiple agents in bulk.

        Args:
            agents: List of tuples, each containing (AgentCard, AgentProvider).
            tag: Optional tag(s) to apply to all agents being added.
                Applied in addition to any tags on individual AgentCards.
            tag_update_strategy: Strategy for updating tags when agents already exist.

        Returns:
            Result[AgentCard, Exception] or list[Result[AgentCard, Exception]]:
                Result object(s) containing the added agent card(s) or exception(s).

        """
        results = []
        for card, agent in agents:
            results.append(self.add_agent(card, agent, tag=tag, tag_update_strategy=tag_update_strategy))
        return results

    def remove_agent(self,
                     *,
                     id: Union[str, list[str]] = None,
                     tag: Optional[Union[Tag, list[Tag]]] = None,
                     tag_match_strategy: TagMatchStrategy = TagMatchStrategy.ALL,
                     skip_if_not_exists: bool = False,
                     ) -> Result[Optional[AgentCard], Exception] | list[Result[Optional[AgentCard], Exception]]:
        """
        Remove agent(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of agents to remove.
            tag: Single tag or list of tags; removes all agents with matching tags.
            tag_match_strategy: Strategy for matching tags when using tag parameter.
            skip_if_not_exists: If True, skip non-existent resources.

        Returns:
            Result[Optional[AgentCard], Exception] or list[Result[Optional[AgentCard], Exception]]:
                Result object(s) containing the removed agent card(s) or exception(s).

        """
        results = []

        ids = []
        if id is not None:
            ids = [id] if isinstance(id, str) else id

        if tag is not None:
            tag_ids = self._tag_mgr.find_resources_by_tags(tag, tag_match_strategy)
            ids = list(set(ids) | set(tag_ids)) if ids else tag_ids

        if not ids:
            return Ok(None)

        for agent_id in ids:
            try:
                if not self._tag_mgr.has_resource(agent_id):
                    if skip_if_not_exists:
                        results.append(Ok(None))
                        continue
                    else:
                        raise JiuWenBaseException(
                            StatusCode.SESSION_AGENT_REMOVE_FAILED.code,
                            StatusCode.SESSION_AGENT_REMOVE_FAILED.errmsg.format(f"Agent '{agent_id}' does not exist.")
                        )
                agent_card = self._id_to_card.get(agent_id, None)
                if not skip_if_not_exists and agent_card is None:
                    raise JiuWenBaseException(
                        StatusCode.SESSION_AGENT_REMOVE_FAILED.code,
                        StatusCode.SESSION_AGENT_REMOVE_FAILED.errmsg.format(f"Agent '{agent_id}' card does not exist.")
                    )
                self._resource_registry.agent().remove_agent(agent_id)
                self._tag_mgr.untag_resource(agent_id)
                results.append(Ok(agent_card))
            except Exception as e:
                results.append(Error(e))

        return results[0] if isinstance(id, str) and tag is None else results

    async def get_agent(self,
                        *,
                        id: Union[str, list[str]] = None,
                        tag: Optional[Union[Tag, list[Tag]]] = None,
                        tag_match_strategy: TagMatchStrategy = TagMatchStrategy.ALL,
                        session: Optional[Session] = None
                        ) -> Optional[BaseAgent] | list[Optional[BaseAgent]]:
        """
        Get agent instance(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of agents to retrieve.
            tag: Single tag or list of tags; returns all agents with matching tags.
            tag_match_strategy: Strategy for matching tags when using tag parameter.
            session: Optional session context for the agents.

        Returns:
            BaseAgent or list[BaseAgent]: Agent instance(s) if found, None otherwise.

        """
        results = []

        ids = []
        if id is not None:
            ids = [id] if isinstance(id, str) else id

        if tag is not None:
            tag_ids = self._tag_mgr.find_resources_by_tags(tag, tag_match_strategy)
            ids = list(set(ids) | set(tag_ids)) if ids else tag_ids

        if not ids:
            return Ok(None)

        for agent_id in ids:
            try:
                if not self._tag_mgr.has_resource(agent_id):
                    raise JiuWenBaseException(
                        StatusCode.SESSION_AGENT_GET_FAILED.code,
                        StatusCode.SESSION_AGENT_GET_FAILED.errmsg.format(f"Agent '{agent_id}' does not exist.")
                    )
                results.append(Ok(self._resource_registry.agent().get_agent(agent_id=agent_id, session=session)))
            except Exception as e:
                results.append(Error(e))

        return results[0] if isinstance(id, str) and tag is None else results

    def add_workflow(self,
                     card: WorkflowCard,
                     workflow: WorkflowProvider,
                     *,
                     tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                     tag_update_strategy: TagUpdateStrategy = TagUpdateStrategy.MERGE
                     ) -> Result[WorkflowCard, Exception]:
        """
        Add a single workflow to the resource manager.

        Args:
            card: The workflow's metadata card containing configuration and identification.
            workflow: Callable provider that creates or returns the workflow instance.
            tag: Optional tag(s) for categorizing and filtering the workflow.
            tag_update_strategy: Strategy for updating tags when workflow already exists.

        Returns:
            Result[WorkflowCard, Exception]: Result object containing the added workflow card or an exception.

        """
        try:
            if not self._resource_registry.is_id_unique(card.id):
                raise JiuWenBaseException(
                    StatusCode.SESSION_WORKFLOW_ADD_FAILED.code,
                    StatusCode.SESSION_WORKFLOW_ADD_FAILED.errmsg.format(
                        reason=f"When registering resource, id should be unique"
                    )
                )
            self._resource_registry.workflow().add_workflow(workflow_id=card.id, workflow=workflow)
            self._tag_mgr.replace_resource_tags(card.id, tag, tag_update_strategy)
            return Ok(card)
        except Exception as e:
            return Error(e)

    def add_workflows(self,
                      workflows: list[Tuple[WorkflowCard, WorkflowProvider]],
                      *,
                      tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                      tag_update_strategy: TagUpdateStrategy = TagUpdateStrategy.MERGE
                      ) -> Result[WorkflowCard, Exception] | list[Result[WorkflowCard, Exception]]:
        """
        Add multiple workflows in bulk.

        Args:
            workflows: List of tuples, each containing (WorkflowCard, WorkflowProvider).
            tag: Optional tag(s) to apply to all workflows being added.
            tag_update_strategy: Strategy for updating tags when workflows already exist.

        Returns:
            Result[WorkflowCard, Exception] or list[Result[WorkflowCard, Exception]]:
                Result object(s) containing the added workflow card(s) or exception(s).

        """
        results = []
        for card, workflow in workflows:
            results.append(self.add_workflow(card, workflow, tag=tag, tag_update_strategy=tag_update_strategy))
        return results

    def remove_workflow(self,
                        *,
                        id: Union[str, list[str]] = None,
                        tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                        tag_match_strategy: TagMatchStrategy = TagMatchStrategy.ALL,
                        skip_if_not_exists: bool = False,
                        ) -> Result[Optional[WorkflowCard], Exception] | list[
        Result[Optional[WorkflowCard], Exception]]:
        """
        Remove workflow(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of workflows to remove.
            tag: Single tag or list of tags; removes all workflows with matching tags.
            tag_match_strategy: Strategy for matching tags when using tag parameter.
            skip_if_not_exists: If True, skip non-existent workflows.

        Returns:
            Result[Optional[WorkflowCard], Exception] or list[Result[Optional[WorkflowCard], Exception]]:
                Result object(s) containing the removed workflow card(s) or exception(s).

        """
        results = []

        ids = []
        if id is not None:
            ids = [id] if isinstance(id, str) else id

        if tag is not None:
            tag_ids = self._tag_mgr.find_resources_by_tags(tag, tag_match_strategy)
            ids = list(set(ids) | set(tag_ids)) if ids else tag_ids

        if not ids:
            return Ok(None)

        for workflow_id in ids:
            try:
                if not self._tag_mgr.has_resource(workflow_id):
                    if skip_if_not_exists:
                        results.append(Ok(None))
                        continue
                    else:
                        raise JiuWenBaseException(
                            StatusCode.SESSION_WORKFLOW_REMOVE_FAILED.code,
                            StatusCode.SESSION_WORKFLOW_REMOVE_FAILED.errmsg.format(
                                f"Workflow '{workflow_id}' does not exist.")
                        )
                workflow_card = self._id_to_card.get(workflow_id, None)
                if not skip_if_not_exists and workflow_card is None:
                    raise JiuWenBaseException(
                        StatusCode.SESSION_WORKFLOW_REMOVE_FAILED.code,
                        StatusCode.SESSION_WORKFLOW_REMOVE_FAILED.errmsg.format(
                            f"Workflow '{workflow_id}' card does not exist.")
                    )
                self._resource_registry.workflow().remove_workflow(workflow_id)
                self._tag_mgr.untag_resource(workflow_id)
                results.append(Ok(workflow_card))
            except Exception as e:
                results.append(Error(e))

        return results[0] if isinstance(id, str) and tag is None else results

    async def get_workflow(self,
                           *,
                           id: Union[str, list[str]] = None,
                           tag: Optional[Union[Tag, list[Tag]]] = None,
                           tag_match_strategy: TagMatchStrategy = TagMatchStrategy.ALL,
                           session: Optional[Session] = None
                           ) -> Optional[Workflow] | list[Optional[Workflow]]:
        """
        Get workflow instance(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of workflows to retrieve.
            tag: Single tag or list of tags; returns all workflows with matching tags.
            tag_match_strategy: Strategy for matching tags when using tag parameter.
            session: Optional session context for the workflows.

        Returns:
            Workflow or list[Workflow]: Workflow instance(s) if found, None otherwise.
        """
        results = []

        ids = []
        if id is not None:
            ids = [id] if isinstance(id, str) else id

        if tag is not None:
            tag_ids = self._tag_mgr.find_resources_by_tags(tag, tag_match_strategy)
            ids = list(set(ids) | set(tag_ids)) if ids else tag_ids

        if not ids:
            return None

        for workflow_id in ids:
            try:
                if not self._tag_mgr.has_resource(workflow_id):
                    raise JiuWenBaseException(
                        StatusCode.SESSION_WORKFLOW_GET_FAILED.code,
                        StatusCode.SESSION_WORKFLOW_GET_FAILED.errmsg.format(
                            f"Workflow '{workflow_id}' does not exist.")
                    )
                results.append(
                    self._resource_registry.workflow().get_workflow(workflow_id=workflow_id, session=session))
            except Exception as e:
                results.append(e)

        return results[0] if isinstance(id, str) and tag is None else results

    def add_tool(self,
                 tool: Union[Tool, list[Tool]],
                 *,
                 tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                 tag_update_strategy: TagUpdateStrategy = TagUpdateStrategy.MERGE
                 ) -> Result[ToolCard, Exception] | list[Result[ToolCard, Exception]]:
        """
        Add tool(s) to the resource manager.

        Args:
            tool: Single Tool instance or list of Tool instances to add.
            tag: Optional tag(s) for categorizing and filtering the tool(s).
            tag_update_strategy: Strategy for updating tags when tools already exist.

        Returns:
            Result[ToolCard, Exception] or list[Result[ToolCard, Exception]]:
                Result object(s) containing the added tool card(s) or exception(s).
        """
        tools = []
        results = []
        if isinstance(tool, Tool):
            tools = [tool]
        elif isinstance(tool, list):
            tools = [item for item in tool]

        for _tool in tools:
            try:
                if not self._resource_registry.is_id_unique(_tool.card().id):
                    raise JiuWenBaseException(
                        StatusCode.SESSION_TOOL_ADD_FAILED.code,
                        StatusCode.SESSION_TOOL_ADD_FAILED.errmsg.format(
                            reason=f"When registering resource, id should be unique"
                        )
                    )
                self._resource_registry.tool().add_tool(_tool.card().id, tool)
                self._tag_mgr.replace_resource_tags(_tool.card().id, tag, tag_update_strategy)
                self._id_to_card[_tool.card().id] = _tool.card()
                results.append(Ok(_tool.card()))
            except Exception as e:
                results.append(Error(e))
        return results

    def get_tool(self,
                 *,
                 id: Union[str, list[str]] = None,
                 tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                 tag_match_strategy: TagMatchStrategy = TagMatchStrategy.ALL,
                 session: Optional[Session] = None
                 ) -> Optional[Tool] | list[Optional[Tool]]:
        """
        Get tool(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of tools to retrieve.
            tag: Single tag or list of tags; returns all tools with matching tags.
            tag_match_strategy: Strategy for matching tags when using tag parameter.
            session: Optional session context for the tools.

        Returns:
            Tool or list[Tool]: Tool instance(s) if found, None otherwise.
        """
        results = []

        ids = []
        if id is not None:
            ids = [id] if isinstance(id, str) else id

        if tag is not None:
            tag_ids = self._tag_mgr.find_resources_by_tags(tag, tag_match_strategy)
            ids = list(set(ids) | set(tag_ids)) if ids else tag_ids

        if not ids:
            return Ok(None)

        for tool_id in ids:
            try:
                if not self._tag_mgr.has_resource(tool_id):
                    raise JiuWenBaseException(
                        StatusCode.SESSION_TOOL_GET_FAILED.code,
                        StatusCode.SESSION_TOOL_GET_FAILED.errmsg.format(f"Tool '{tool_id}' does not exist.")
                    )
                results.append(Ok(self._resource_registry.tool().get_tool(tool_id=tool_id, session=session)))
            except Exception as e:
                results.append(Error(e))

        return results[0] if isinstance(id, str) and tag is None else results

    def remove_tool(self,
                    *,
                    id: Union[str, list[str]] = None,
                    tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                    tag_match_strategy: TagMatchStrategy = TagMatchStrategy.ALL,
                    skip_if_not_exists: bool = False,
                    ) -> Result[Optional[ToolCard], Exception] | list[Result[Optional[ToolCard], Exception]]:
        """
        Remove tool(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of tools to remove.
            tag: Single tag or list of tags; removes all tools with matching tags.
            tag_match_strategy: Strategy for matching tags when using tag parameter.
            skip_if_not_exists: If True, skip non-existent tools.

        Returns:
            Result[Optional[ToolCard], Exception] or list[Result[Optional[ToolCard], Exception]]:
                Result object(s) containing the removed tool card(s) or exception(s).
        """
        results = []

        ids = []
        if id is not None:
            ids = [id] if isinstance(id, str) else id

        if tag is not None:
            tag_ids = self._tag_mgr.find_resources_by_tags(tag, tag_match_strategy)
            ids = list(set(ids) | set(tag_ids)) if ids else tag_ids

        if not ids:
            return Ok(None)

        for tool_id in ids:
            try:
                if not self._tag_mgr.has_resource(tool_id):
                    if skip_if_not_exists:
                        results.append(Ok(None))
                        continue
                    else:
                        raise JiuWenBaseException(
                            StatusCode.SESSION_TOOL_REMOVED_FAILED.code,
                            StatusCode.SESSION_TOOL_REMOVED_FAILED.errmsg.format(
                                f"Tool '{tool_id}' does not exist.")
                        )
                tool_card = self._id_to_card.get(tool_id, None)
                if not skip_if_not_exists and tool_card is None:
                    raise JiuWenBaseException(
                        StatusCode.SESSION_TOOL_REMOVED_FAILED.code,
                        StatusCode.SESSION_TOOL_REMOVED_FAILED.errmsg.format(
                            f"Tool '{tool_id}' card does not exist.")
                    )
                self._resource_registry.tool().remove_tool(tool_id)
                self._tag_mgr.untag_resource(tool_id)
                results.append(Ok(tool_card))
            except Exception as e:
                results.append(Error(e))

        return results[0] if isinstance(id, str) and tag is None else results

    def add_model(self,
                  id: str,
                  model: ModelProvider,
                  *,
                  tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                  tag_update_strategy: TagUpdateStrategy = TagUpdateStrategy.MERGE
                  ) -> Result[str, Exception]:
        """
        Add a model to the resource manager.

        Args:
            id: Unique identifier for the model.
            model: Callable provider that creates or returns the model instance.
            tag: Optional tag(s) for categorizing and filtering the model.
            tag_update_strategy: Strategy for updating tags when model already exists.

        Returns:
            Result[str, Exception]: Result object containing the model ID or an exception.

        """
        try:
            if not self._resource_registry.is_id_unique(id):
                raise JiuWenBaseException(
                    StatusCode.SESSION_MODEL_ADD_FAILED.code,
                    StatusCode.SESSION_MODEL_ADD_FAILED.errmsg.format(
                        reason=f"When registering resource, id should be unique"
                    )
                )
            self._resource_registry.model().add_model(model_id=id, model=model)
            self._tag_mgr.replace_resource_tags(id, tag, tag_update_strategy)
            return Ok(id)
        except Exception as e:
            return Error(e)

    def add_models(self,
                   models: list[Tuple[str, ModelProvider]],
                   *,
                   tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                   tag_update_strategy: TagUpdateStrategy = TagUpdateStrategy.MERGE
                   ) -> Result[str, Exception] | list[Result[str, Exception]]:
        """
        Add multiple models in bulk.

        Args:
            models: List of tuples, each containing (model_id, ModelProvider).
            tag: Optional tag(s) to apply to all models being added.
            tag_update_strategy: Strategy for updating tags when models already exist.

        Returns:
            Result[str, Exception] or list[Result[str, Exception]]:
                Result object(s) containing the model ID(s) or exception(s).
        """
        results = []
        for id, model in models:
            results.append(self.add_model(id=id, model=model, tag=tag, tag_update_strategy=tag_update_strategy))
        return results

    def remove_model(self,
                     *,
                     id: Union[str, list[str]] = None,
                     tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                     tag_match_strategy: TagMatchStrategy = TagMatchStrategy.ALL,
                     skip_if_not_exists: bool = False,
                     ) -> Result[str, Exception] | list[Result[str, Exception]]:
        """
        Remove model(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of models to remove.
            tag: Single tag or list of tags; removes all models with matching tags.
            tag_match_strategy: Strategy for matching tags when using tag parameter.
            skip_if_not_exists: If True, skip non-existent models.

        Returns:
            Result[str, Exception] or list[Result[str, Exception]]:
                Result object(s) containing the model ID(s) or exception(s).
        """
        results = []

        ids = []
        if id is not None:
            ids = [id] if isinstance(id, str) else id

        if tag is not None:
            tag_ids = self._tag_mgr.find_resources_by_tags(tag, tag_match_strategy)
            ids = list(set(ids) | set(tag_ids)) if ids else tag_ids

        if not ids:
            if not skip_if_not_exists:
                return Error(
                    JiuWenBaseException(
                        StatusCode.SESSION_MODEL_REMOVED_FAILED.code,
                        StatusCode.SESSION_MODEL_REMOVED_FAILED.errmsg.format(
                            f"The corresponding model does not exist.")
                    )
                )
            return Ok(None)

        for model_id in ids:
            try:
                if not self._tag_mgr.has_resource(model_id):
                    if skip_if_not_exists:
                        results.append(Ok(None))
                        continue
                    else:
                        raise JiuWenBaseException(
                            StatusCode.SESSION_MODEL_REMOVED_FAILED.code,
                            StatusCode.SESSION_MODEL_REMOVED_FAILED.errmsg.format(
                                f"Model '{model_id}' does not exist.")
                        )
                self._resource_registry.model().remove_model(model_id=model_id)
                self._tag_mgr.untag_resource(model_id)
                results.append(Ok(id))
            except Exception as e:
                results.append(Error(e))

        return results[0] if isinstance(id, str) and tag is None else results

    async def get_model(self,
                        *,
                        id: Union[str, list[str]] = None,
                        tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                        tag_match_strategy: TagMatchStrategy = TagMatchStrategy.ALL,
                        session: Optional[Session] = None) \
            -> Optional[BaseModel] | list[Optional[BaseModel]]:
        """
        Get model instance(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of models to retrieve.
            tag: Single tag or list of tags; returns all models with matching tags.
            tag_match_strategy: Strategy for matching tags when using tag parameter.
            session: Optional session context for the models.

        Returns:
            Model or list[Model]: Model instance(s) if found, None otherwise.

        """
        results = []

        ids = []
        if id is not None:
            ids = [id] if isinstance(id, str) else id

        if tag is not None:
            tag_ids = self._tag_mgr.find_resources_by_tags(tag, tag_match_strategy)
            ids = list(set(ids) | set(tag_ids)) if ids else tag_ids

        if not ids:
            return None

        for model_id in ids:
            try:
                if not self._tag_mgr.has_resource(model_id):
                    raise JiuWenBaseException(
                        StatusCode.SESSION_MODEL_GET_FAILED.code,
                        StatusCode.SESSION_MODEL_GET_FAILED.errmsg.format(f"Model '{model_id}' does not exist.")
                    )
                results.append(self._resource_registry.model().get_model(model_id=model_id, session=session))
            except Exception as e:
                results.append(e)

        return results[0] if isinstance(id, str) and tag is None else results

    def add_prompt(self,
                   id: str,
                   template: PromptTemplate,
                   *,
                   tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                   tag_update_strategy: TagUpdateStrategy = TagUpdateStrategy.MERGE
                   ) -> Result[str, Exception]:
        """
        Add a prompt template to the resource manager.

        Args:
            id: Unique identifier for the prompt template.
            template: PromptTemplate instance containing the prompt content and configuration.
            tag: Optional tag(s) for categorizing and filtering the prompt.
            tag_update_strategy: Strategy for updating tags when prompt already exists.

        Returns:
            Result[str, Exception]: Result object containing the prompt ID or an exception.
        """
        try:
            if not self._resource_registry.is_id_unique(id):
                raise JiuWenBaseException(
                    StatusCode.SESSION_PROMPT_ADD_FAILED.code,
                    StatusCode.SESSION_PROMPT_ADD_FAILED.errmsg.format(
                        reason=f"When registering resource, id should be unique"
                    )
                )
            self._resource_registry.prompt().add_prompt(template_id=id, template=template)
            self._tag_mgr.replace_resource_tags(id, tag, tag_update_strategy)
            return Ok(id)
        except Exception as e:
            return Error(e)

    def add_prompts(self,
                    prompts: list[Tuple[str, PromptTemplate]],
                    *,
                    tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                    tag_update_strategy: TagUpdateStrategy = TagUpdateStrategy.MERGE
                    ) -> Result[str, Exception] | list[Result[str, Exception]]:
        """
        Add multiple prompt templates in bulk.

        Args:
            prompts: List of tuples, each containing (prompt_id, PromptTemplate).
            tag: Optional tag(s) to apply to all prompts being added.
            tag_update_strategy: Strategy for updating tags when prompts already exist.

        Returns:
            Result[str, Exception] or list[Result[str, Exception]]:
                Result object(s) containing the prompt ID(s) or exception(s).

        """
        result = []
        for id, prompt in prompts:
            result.append(self.add_prompt(id, prompt, tag=tag, tag_update_strategy=tag_update_strategy))
        return result

    def remove_prompt(self,
                      *,
                      id: Union[str, list[str]] = None,
                      tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                      tag_match_strategy: TagMatchStrategy = TagMatchStrategy.ALL,
                      skip_if_not_exists: bool = False,
                      ) -> Result[str, Exception] | list[Result[str, Exception]]:
        """
        Remove prompt template(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of prompts to remove.
            tag: Single tag or list of tags; removes all prompts with matching tags.
            tag_match_strategy: Strategy for matching tags when using tag parameter.
            skip_if_not_exists: If True, skip non-existent prompts.

        Returns:
            Result[str, Exception] or list[Result[str, Exception]]:
                Result object(s) containing the prompt ID(s) or exception(s).
        """
        results = []

        ids = []
        if id is not None:
            ids = [id] if isinstance(id, str) else id

        if tag is not None:
            tag_ids = self._tag_mgr.find_resources_by_tags(tag, tag_match_strategy)
            ids = list(set(ids) | set(tag_ids)) if ids else tag_ids

        if not ids:
            if not skip_if_not_exists:
                return Error(
                    JiuWenBaseException(
                        StatusCode.SESSION_PROMPT_REMOVED_FAILED.code,
                        StatusCode.SESSION_PROMPT_REMOVED_FAILED.errmsg.format(
                            f"The corresponding prompt does not exist.")
                    )
                )
            return Ok(None)

        for template_id in ids:
            try:
                if not self._tag_mgr.has_resource(template_id):
                    if skip_if_not_exists:
                        results.append(Ok(None))
                        continue
                    else:
                        raise JiuWenBaseException(
                            StatusCode.SESSION_PROMPT_REMOVED_FAILED.code,
                            StatusCode.SESSION_PROMPT_REMOVED_FAILED.errmsg.format(
                                f"Prompt '{template_id}' does not exist.")
                        )
                self._resource_registry.prompt().remove_prompt(template_id=template_id)
                self._tag_mgr.untag_resource(template_id)
                results.append(Ok(id))
            except Exception as e:
                results.append(Error(e))

        return results[0] if isinstance(id, str) and tag is None else results

    def get_prompt(self,
                   *,
                   id: Union[str, list[str]] = None,
                   tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                   tag_match_strategy: TagMatchStrategy = TagMatchStrategy.ALL,
                   ) -> Optional[PromptTemplate] | list[Optional[PromptTemplate]]:
        """
        Get prompt template(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of prompts to retrieve.
            tag: Single tag or list of tags; returns all prompts with matching tags.
            tag_match_strategy: Strategy for matching tags when using tag parameter.

        Returns:
            PromptTemplate or list[PromptTemplate]: Prompt template instance(s) if found, None otherwise.
        """
        results = []

        ids = []
        if id is not None:
            ids = [id] if isinstance(id, str) else id

        if tag is not None:
            tag_ids = self._tag_mgr.find_resources_by_tags(tag, tag_match_strategy)
            ids = list(set(ids) | set(tag_ids)) if ids else tag_ids

        if not ids:
            return None

        for template_id in ids:
            try:
                if not self._tag_mgr.has_resource(template_id):
                    raise JiuWenBaseException(
                        StatusCode.SESSION_PROMPT_GET_FAILED.code,
                        StatusCode.SESSION_PROMPT_GET_FAILED.errmsg.format(f"Prompt '{template_id}' does not exist.")
                    )
                results.append(self._resource_registry.prompt().get_prompt(template_id=id))
            except Exception as e:
                results.append(e)

        return results[0] if isinstance(id, str) and tag is None else results

    async def get_tool_infos(self,
                             *,
                             id: Union[str, list[str]] = None,
                             type: Union[str, list[str]] = None,
                             tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                             tag_match_strategy: TagMatchStrategy = TagMatchStrategy.ALL,
                             ignore_exception: bool = False,
                             ) -> Optional[ToolInfo] | list[Optional[ToolInfo]]:
        """
        Get tool information/metadata by ID, type, or tag.

        Args:
            id: Single ID or list of IDs of tools to get info for.
            type: Single type or list of types to filter tools by.
                Common types: ["function", "mcp", "workflow"].
            tag: Single tag or list of tags; returns info for all tools with matching tags.
            tag_match_strategy: Strategy for matching tags when using tag parameter.
            ignore_exception: If True, ignore exceptions and return None for failing items.

        Returns:
            ToolInfo or list[ToolInfo]: Tool information instance(s) if found, None otherwise.
        """
        results = []

        ids = []
        if id is not None:
            ids = [id] if isinstance(id, str) else id

        if tag is not None:
            tag_ids = self._tag_mgr.find_resources_by_tags(tag, tag_match_strategy)
            ids = list(set(ids) | set(tag_ids)) if ids else tag_ids

        if not ids:
            return None

        types = []
        if type is not None:
            types = [type] if isinstance(type, str) else type

        for tool_id in ids:
            try:
                if not self._tag_mgr.has_resource(tool_id):
                    raise JiuWenBaseException(
                        StatusCode.SESSION_TOOL_TOOL_INFO_GET_FAILED.code,
                        StatusCode.SESSION_TOOL_TOOL_INFO_GET_FAILED.errmsg.format(
                            f"Tool info '{tool_id}' does not exist.")
                    )
                tool_info = self._resource_registry.tool().get_tool_infos(tool_ids=tool_id)
                if not tool_info:
                    tool_info = self._resource_registry.workflow().get_tool_infos(workflow_ids=tool_id)
                if types and tool_info.type not in types:
                    continue
                results.append(tool_info)
            except Exception as e:
                if ignore_exception:
                    results.append(None)
                    continue
                results.append(e)

        return results[0] if isinstance(id, str) and tag is None else results

    async def add_mcp_server(self,
                             server_config: Union[McpServerConfig, list[McpServerConfig]],
                             *,
                             tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                             tag_update_strategy: TagUpdateStrategy = TagUpdateStrategy.MERGE,
                             expiry_time: Optional[float] = None,
                             ignore_exception: bool = False
                             ) -> Result[str, Exception] | list[Result[str, Exception]]:
        """
        Add MCP (Model Context Protocol) server configuration(s).

        Args:
            server_config: Single or list of McpServerConfig instances.
            tag: Optional tag(s) for categorizing the server(s).
            tag_update_strategy: Strategy for updating tags when servers already exist.
            expiry_time: Optional Unix timestamp when the server configuration expires.
                If None, the configuration does not expire.
            ignore_exception: If True, continue adding other servers if one fails.

        Returns:
            Result[str, Exception] or list[Result[str, Exception]]:
                Result object(s) containing the server name(s) or exception(s).
        """
        server_configs = []
        add_results = []
        if isinstance(server_config, McpServerConfig):
            server_configs = [server_config]
        elif isinstance(server_config, list):
            server_configs = [item for item in server_config]

        for _tool in server_configs:
            try:
                if not self._tag_mgr.has_resource(server_config.server_name):
                    raise JiuWenBaseException(
                        StatusCode.SESSION_MCP_SERVER_ADD_FAILED.code,
                        StatusCode.SESSION_MCP_SERVER_ADD_FAILED.errmsg.format(
                            f"Mcp server '{server_config.server_name}' existS.")
                    )
                results = await self._resource_registry.tool().add_tool_servers(server_config)
                for config, result in zip(server_config, results):
                    if result:
                        add_results.append(Ok(config.server_name))
                        self._tag_mgr.replace_resource_tags(config.server_name, tag, tag_update_strategy)
                    else:
                        add_results.append(
                            Error(
                                JiuWenBaseException(
                                    StatusCode.SESSION_MCP_SERVER_ADD_FAILED.code,
                                    StatusCode.SESSION_MCP_SERVER_ADD_FAILED.errmsg.format(
                                        f"{str(result)}")
                                )))
            except Exception as e:
                add_results.append(Error(e))
        return add_results

    async def refresh_mcp_server(self,
                                 server_name: Union[str, list[str]],
                                 *,
                                 tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                                 tag_match_strategy: TagMatchStrategy = TagMatchStrategy.ALL,
                                 ignore_exception: bool = False,
                                 skip_if_not_exists: bool = False,
                                 ) -> Result[str, Exception] | list[Result[str, Exception]]:
        """
        Refresh MCP server toolinfos(s) by name.

        Args:
            server_name: Single or list of MCP server names to refresh.
            tag: Optional tag to filter servers to refresh.
            tag_match_strategy: Strategy for matching tags when using tag parameter.
            ignore_exception: If True, continue refreshing other servers if one fails.
            skip_if_not_exists: If True, skip non-existent servers.

        Returns:
            Result[str, Exception] or list[Result[str, Exception]]:
                Result object(s) containing the server name(s) or exception(s).
        """
        results = []

        server_names = []
        if server_name is not None:
            server_names = [server_name] if isinstance(server_name, str) else server_name

        if tag is not None:
            tag_ids = self._tag_mgr.find_resources_by_tags(tag, tag_match_strategy)
            ids = list(set(server_names) | set(tag_ids)) if server_names else tag_ids

        if not ids:
            return None

        for _server_name in ids:
            try:
                if not self._tag_mgr.has_resource(_server_name):
                    raise JiuWenBaseException(
                        StatusCode.SESSION_TOOL_TOOL_INFO_GET_FAILED.code,
                        StatusCode.SESSION_TOOL_TOOL_INFO_GET_FAILED.errmsg.format(
                            f"Tool info '{_server_name}' does not exist.")
                    )
                # todo mcp refresh server
            except Exception as e:
                results.append(e)

        return results[0] if isinstance(id, str) and tag is None else results

    async def remove_mcp_server(self,
                                *,
                                server_name: Optional[Union[str, list[str]]] = None,
                                tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                                tag_match_strategy: TagMatchStrategy = TagMatchStrategy.ALL,
                                skip_if_not_exists: bool = False,
                                ignore_exception: bool = False,
                                ) -> Result[str, Exception] | list[Result[str, Exception]]:
        """
        Remove MCP server(s) by name or tag.

        Args:
            server_name: Single or list of MCP server names to remove.
            tag: Single tag or list of tags; removes all servers with matching tags.
            tag_match_strategy: Strategy for matching tags when using tag parameter.
            skip_if_not_exists: If True, skip non-existent servers.
            ignore_exception: If True, continue removing other servers if one fails.

        Returns:
            Result[str, Exception] or list[Result[str, Exception]]:
                Result object(s) containing the server name(s) or exception(s).
        """
        results = []

        server_names = []
        if server_name is not None:
            server_names = [server_name] if isinstance(server_name, str) else server_name

        if tag is not None:
            tag_ids = self._tag_mgr.find_resources_by_tags(tag, tag_match_strategy)
            server_names = list(set(server_names) | set(tag_ids)) if server_names else tag_ids

        if not server_names:
            if not skip_if_not_exists:
                return Error(
                    JiuWenBaseException(
                        StatusCode.SESSION_MCP_SERVER_REMOVED_FAILED.code,
                        StatusCode.SESSION_MCP_SERVER_REMOVED_FAILED.errmsg.format(
                            f"The corresponding mcp server does not exist.")
                    )
                )
            return Ok(None)

        for _server_name in server_names:
            try:
                if not self._tag_mgr.has_resource(_server_name):
                    if skip_if_not_exists:
                        results.append(Ok(None))
                        continue
                    else:
                        raise JiuWenBaseException(
                            StatusCode.SESSION_MCP_SERVER_REMOVED_FAILED.code,
                            StatusCode.SESSION_MCP_SERVER_REMOVED_FAILED.errmsg.format(
                                f"Prompt '{_server_name}' does not exist.")
                        )
                removed_tool_servers = await self._resource_registry.tool().remove_tool_server(server_name)
                removed_results = []
                for name, _result in zip(server_name, removed_tool_servers):
                    if _result:
                        removed_results.append(Ok(name))
                    else:
                        if skip_if_not_exists:
                            continue
                        removed_results.append(Error(
                            JiuWenBaseException(StatusCode.SESSION_MCP_SERVER_REMOVED_FAILED.code,
                                                StatusCode.SESSION_MCP_SERVER_REMOVED_FAILED.errmsg.format(
                                                    f"remove mcp server {name} failed"))))
                self._tag_mgr.untag_resource(_server_name)
                results.append(Ok(id))
            except Exception as e:
                results.append(Error(e))

        return results[0] if isinstance(id, str) and tag is None else results

    async def get_mcp_tool(self,
                           *,
                           name: Union[str, list[str]],
                           server_name: Union[str, list[str]],
                           tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                           tag_match_strategy: TagMatchStrategy = TagMatchStrategy.ALL,
                           session: Optional[Session] = None
                           ) -> Optional[Tool] | list[Optional[Tool]]:
        """
        Get MCP tool(s) by name and server.

        Args:
            name: Single or list of MCP tool names to retrieve.
            server_name: Single or list of MCP server names containing the tools.
            tag: Optional tag to filter servers/tools.
            tag_match_strategy: Strategy for matching tags when using tag parameter.
            session: Optional session context for the tools.

        Returns:
            Tool or list[Tool]: MCP tool instance(s) if found, None otherwise.
        """
        pass

    async def get_mcp_tool_infos(self,
                                 *,
                                 name: Union[str, list[str]] = None,
                                 server_name: Union[str, list[str]] = None,
                                 tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                                 tag_match_strategy: TagMatchStrategy = TagMatchStrategy.ALL,
                                 ignore_exception: bool = False,
                                 ) -> Optional[ToolInfo] | list[Optional[ToolInfo]]:
        """
        Get MCP tool information/metadata by name and server.

        Args:
            name: Single or list of MCP tool names to get info for.
                If None, returns info for all tools in specified servers.
            server_name: Single or list of MCP server names containing the tools.
                Must be provided if name is None.
            tag: Optional tag to filter servers/tools.
            tag_match_strategy: Strategy for matching tags when using tag parameter.
            ignore_exception: If True, ignore exceptions for individual items.

        Returns:
            ToolInfo or list[ToolInfo]: MCP tool information instance(s) if found, None otherwise.
        """
        if not server_name:
            return None
        tool_mgr = self._resource_registry.tool()
        single = isinstance(server_name, str)
        names = [server_name] if single else server_name
        results = [tool_mgr.get_tool_infos(tool_server_name=n) for n in names]
        return results[0] if single else results

    def get_resource_by_tag(self,
                            tag: Tag) -> Optional[list[BaseCard]]:
        """
        Retrieve all resources associated with a specific tag.

        Args:
            tag: Tag to search for.

        Returns:
            List of BaseCard instances representing resources with the specified tag,
            or None if no resources found.
        """
        ids = self._tag_mgr.find_resources_by_tags(tag, TagMatchStrategy.ANY)
        if not ids:
            return None
        return [card for _, card in self._id_to_card.items() if _ in ids]

    def list_tags(self) -> list[Tag]:
        """
        List all tags currently in use across all resources.

        Returns:
            List of unique tag strings.
        """
        return self._tag_mgr.has_tags()

    def has_tag(self, tag: str) -> bool:
        """
            Check if the specified tag exists in the resource_mgr.
        """
        return self._tag_mgr.has_tag(tag)

    async def remove_tag(self,
                         tag: Union[Tag, list[Tag]] = None,
                         ignore_exception: bool = False,
                         *,
                         ignore_if_not_exists: bool = False,
                         ) -> Result[Tag, Exception] | list[Result[Tag, Exception]]:
        """
        Remove tag(s) from all resources.

        Args:
            tag: Single tag or list of tags to remove from all resources.
            ignore_exception: If True, ignore exceptions during removal.
            ignore_if_not_exists: If True, ignore non-existent tags.

        Returns:
            Result[Tag, Exception] or list[Result[Tag, Exception]]:
                Result object(s) containing the tag(s) or exception(s).
        """
        remove_result = []
        tags = list(tag) if isinstance(tag, Tag) else tag
        for _tag in tags:
            if not self._tag_mgr.has_tag(_tag) and not ignore_if_not_exists:
                result = Error(
                    JiuWenBaseException(
                        StatusCode.SESSION_TAG_MANAGE_FAILED.code,
                        StatusCode.SESSION_TAG_MANAGE_FAILED.errmsg.format(
                            reason=f"Remove specific tag from a resource error, non-existent tag: {_tag}."
                        )
                    ))
                remove_result.append(result)
                continue

            try:
                ids = self._tag_mgr.find_resources_by_tags(_tag, TagMatchStrategy.ANY)
                for id in ids:
                    self._tag_mgr.remove_resource_tags(id, _tag)
                remove_result.append(Ok(_tag))
            except Exception as e:
                if not ignore_exception:
                    result = Error(e)
                    remove_result.append(result)

        return remove_result

    def update_resource_tag(self,
                            id: str,
                            tag: Union[Tag, list[Tag]]
                            ) -> Result[list[Tag], Exception]:
        """
        Replace all tags on a resource with new tag(s).

        Args:
            id: Resource identifier.
            tag: New tag(s) to set on the resource.

        Returns:
            Result[list[Tag], Exception]: Result object containing the new tag list or an exception.

        """
        try:
            self._tag_mgr.replace_resource_tags(id, tag, TagUpdateStrategy.REPLACE)
            return Ok(self._tag_mgr.get_resources_tags(id))
        except Exception as e:
            return Error(e)

    def add_resource_tag(self,
                         id: str,
                         tag: Union[Tag, list[Tag]]
                         ) -> Result[list[Tag], Exception]:
        """
        Add tag(s) to a resource.

        Args:
            id: Resource identifier.
            tag: Tag(s) to add to the resource.

        Returns:
            Result[list[Tag], Exception]: Result object containing all tags now associated with the resource.
        """
        try:
            self._tag_mgr.tag_resource(id, tag)
            return Ok(self._tag_mgr.get_resources_tags(id))
        except Exception as e:
            return Error(e)

    def remove_resource_tag(self,
                            id: str,
                            tag: Union[Tag, list[Tag]],
                            *,
                            ignore_if_not_exists: bool = False
                            ) -> Result[list[Tag], Exception]:
        """
        Remove specific tag(s) from a resource.

        Args:
            id: Resource identifier.
            tag: Tag(s) to remove from the resource.
            ignore_if_not_exists: If True, ignore non-existent tags.

        Returns:
            Result[list[Tag], Exception]: Result object containing remaining tags on the resource.
        """
        existed_tags = self._tag_mgr.get_resources_tags(id)
        if isinstance(tag, Tag):
            tag = [tag]
        existing_tags_to_remove = set(tag) & set(existed_tags)
        non_existing_tags = set(tag) - set(existed_tags)

        if non_existing_tags and not ignore_if_not_exists:
            return Error(
                JiuWenBaseException(
                    StatusCode.SESSION_TAG_MANAGE_FAILED.code,
                    StatusCode.SESSION_TAG_MANAGE_FAILED.errmsg.format(
                        reason=f"Remove specific tag(s) from a resource error, non-existent tag(s): {non_existing_tags}."
                    )
                )
            )
        self._tag_mgr.remove_resource_tags(id, list(existing_tags_to_remove))
        return Ok(self._tag_mgr.get_resources_tags(id))

    def get_resource_tag(self, id: str) -> Optional[list[Tag]]:
        """
        Get all tags associated with a resource.

        Args:
            id: Resource identifier.

        Returns:
            List of tags associated with the resource, or None if resource not found.
        """
        resource_tag = self._tag_mgr.get_resources_tags(id)
        return resource_tag if resource_tag else None

    def resource_has_tag(self, id: str, tag: Tag) -> bool:
        """
        Check if a specific resource is associated with the given tag.

        Args:
            id: The unique identifier of the resource to check.
            tag: The tag to verify association with the resource.
        Returns:
            True if the resource has the specified tag, False otherwise.
        """
        return tag in self._tag_mgr.get_resources_tags(id)

    async def release(self):
        await self._resource_registry.tool().release()
