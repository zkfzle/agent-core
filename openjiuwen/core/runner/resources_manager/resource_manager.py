#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from openjiuwen.core.foundation.tool import Tool, ToolInfo, ToolCard
from openjiuwen.core.multi_agent import BaseGroup, GroupCard
from openjiuwen.core.protocols.mcp import McpServerConfig
from openjiuwen.core.runner.resources_manager.base import AgentGroupProvider, Tag, GLOBAL, Result, TagUpdateStrategy, \
    TagMatchStrategy, AgentProvider, WorkflowProvider, ModelProvider
from openjiuwen.core.runner.resources_manager.tool_manager import ToolMgr
from openjiuwen.core.runner.resources_manager.workflow_manager import WorkflowMgr
from openjiuwen.core.runner.resources_manager.prompt_manager import PromptMgr
from openjiuwen.core.runner.resources_manager.model_manager import ModelMgr

from typing import Optional, Union, Tuple

from openjiuwen.core.session import Session
from openjiuwen.core.single_agent import BaseAgent, AgentCard
from openjiuwen.core.workflow.base import WorkflowCard

Workflow = TypeVar("Workflow", contravariant=True)


class ResourceManager(ABC):
    @abstractmethod
    def tool(self) -> ToolMgr:
        pass

    @abstractmethod
    def prompt(self) -> PromptMgr:
        pass

    @abstractmethod
    def model(self) -> ModelMgr:
        pass

    @abstractmethod
    def workflow(self) -> WorkflowMgr:
        pass


class ResourceMgr(ResourceManager):
    """
    Resource Manager for Model, Workflow, Prompt, Tool
    """

    def __init__(self) -> None:
        self._tool_mgr = ToolMgr()
        self._workflow_mgr = WorkflowMgr()
        self._prompt_mgr = PromptMgr()
        self._model_mgr = ModelMgr()

    def tool(self) -> ToolMgr:
        return self._tool_mgr

    def prompt(self) -> PromptMgr:
        return self._prompt_mgr

    def model(self) -> ModelMgr:
        return self._model_mgr

    def workflow(self) -> WorkflowMgr:
        return self._workflow_mgr

    def add_agent_group(self,
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

        Examples:
            >>> # Add new agent group with tags
            >>> result = mgr.add_agent_group(
            ...     card=GroupCard(id="support_team", name="Customer Support"),
            ...     agent_group=create_support_group,
            ...     tag=["customer_service", "priority_high"]
            ... )
            >>> if result.is_ok():
            ...     print(f"Added agent group: {result.value.id}")
            >>> else:
            ...     print(f"Failed to add agent group: {result.exception}")
        """
        pass

    def remove_agent_group(self,
                           *,
                           id: Optional[Union[str, list[str]]] = None,
                           tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                           tag_match_stategy: TagMatchStrategy = TagMatchStrategy.ALL,
                           skip_if_not_exists: bool = False,
                           ) -> Result[Optional[GroupCard], Exception] | list[Result[Optional[GroupCard], Exception]]:
        """
        Remove agent group(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of agent groups to remove.
                Cannot be used together with tag parameter.
            tag: Single tag or list of tags; removes all agent groups with matching tags.
                Cannot be used together with id parameter.
            tag_match_stategy: Strategy for matching tags when using tag parameter.
                ALL - Resource must have all specified tags.
                ANY - Resource must have at least one of the specified tags.
            skip_if_not_exists: If True, silently skip non-existent resources.
                If False, raise ResourceNotFoundError for non-existent resources.

        Returns:
            Result[Optional[GroupCard], Exception] or list[Result[Optional[GroupCard], Exception]]:
                Result object(s) containing the removed group card(s) or exception.

        Examples:
            >>> # Remove by ID
            >>> result = mgr.remove_agent_group(id="support_team")
            >>> if result.is_ok() and result.value:
            ...     print(f"Removed agent group: {result.value.id}")
            >>>
            >>> # Remove by tag
            >>> result = mgr.remove_agent_group(tag="deprecated")
            >>>
            >>> # Remove multiple by ID
            >>> results = mgr.remove_agent_group(id=["team1", "team2"])
            >>> for result in results:
            ...     if result.is_ok() and result.value:
            ...         print(f"Removed: {result.value.id}")
        """
        pass

    async def get_agent_group(self,
                              *,
                              id: str = None,
                              tag: Optional[Tag] = None,
                              tag_match_stategy: TagMatchStrategy = TagMatchStrategy.ALL,
                              session: Optional[Session] = None
                              ) -> Optional[BaseGroup]:
        """
        Get an agent group instance by ID or tag.

        Args:
            id: Unique identifier of the agent group. Either id or tag must be provided.
            tag: Optional tag for filtering when id is provided,
                 or main lookup criteria when id is not provided.
            tag_match_stategy: Strategy for matching tags when using tag parameter.
            session: Optional session context for the agent group.
                If provided, the agent group will be initialized with this session.

        Returns:
            BaseGroup instance if found, None otherwise.

        Raises:
            ValueError: When neither id nor tag is provided.

        Examples:
            >>> # Get by ID
            >>> group = await mgr.get_agent_group(id="support_team")
            >>>
            >>> # Get by tag
            >>> group = await mgr.get_agent_group(tag="customer_service")
            >>>
            >>> # Get by ID with tag filter
            >>> group = await mgr.get_agent_group(
            ...     id="support_team",
            ...     tag="active"
            ... )
        """
        pass

    def add_agent(self,
                  card: AgentCard,
                  agent: AgentProvider,
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

        Examples:
            >>> result = mgr.add_agent(
            ...     card=AgentCard(id="translator", name="Translation Agent"),
            ...     agent=create_translator_agent,
            ...     tag=["language", "utility"]
            ... )
            >>> if result.is_ok():
            ...     print(f"Added agent: {result.value.id}")
        """
        pass

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

        Examples:
            >>> agents = [
            ...     (AgentCard(id="agent1", name="Agent One"), create_agent1),
            ...     (AgentCard(id="agent2", name="Agent Two"), create_agent2),
            ... ]
            >>> results = mgr.add_agents(
            ...     agents,
            ...     tag="batch_upload"
            ... )
            >>> for result in results:
            ...     if result.is_ok():
            ...         print(f"Added agent: {result.value.id}")
        """
        pass

    def remove_agent(self,
                     *,
                     id: Union[str, list[str]] = None,
                     tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                     tag_match_stategy: TagMatchStrategy = TagMatchStrategy.ALL,
                     skip_if_not_exists: bool = False,
                     ) -> Result[Optional[AgentCard], Exception] | list[Result[Optional[AgentCard], Exception]]:
        """
        Remove agent(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of agents to remove.
            tag: Single tag or list of tags; removes all agents with matching tags.
            tag_match_stategy: Strategy for matching tags when using tag parameter.
            skip_if_not_exists: If True, skip non-existent resources.

        Returns:
            Result[Optional[AgentCard], Exception] or list[Result[Optional[AgentCard], Exception]]:
                Result object(s) containing the removed agent card(s) or exception(s).

        Examples:
            >>> # Remove single agent
            >>> result = mgr.remove_agent(id="translator")
            >>> if result.is_ok() and result.value:
            ...     print(f"Removed agent: {result.value.id}")
            >>>
            >>> # Remove multiple agents by tag
            >>> results = mgr.remove_agent(tag=["deprecated", "test_only"])
        """
        pass

    async def get_agent(self,
                        *,
                        id: Union[str, list[str]] = None,
                        tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                        tag_match_stategy: TagMatchStrategy = TagMatchStrategy.ALL,
                        session: Optional[Session] = None
                        ) -> Optional[BaseAgent] | list[Optional[BaseAgent]]:
        """
        Get agent instance(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of agents to retrieve.
            tag: Single tag or list of tags; returns all agents with matching tags.
            tag_match_stategy: Strategy for matching tags when using tag parameter.
            session: Optional session context for the agents.

        Returns:
            BaseAgent or list[BaseAgent]: Agent instance(s) if found, None otherwise.

        Examples:
            >>> # Get single agent
            >>> agent = await mgr.get_agent(id="translator")
            >>>
            >>> # Get multiple agents by tag
            >>> agents = await mgr.get_agent(tag="customer_service")
            >>>
            >>> # Get multiple agents by ID
            >>> agents = await mgr.get_agent(id=["agent1", "agent2"])
        """
        pass

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

        Examples:
            >>> result = mgr.add_workflow(
            ...     card=WorkflowCard(id="onboarding", name="User Onboarding"),
            ...     workflow=create_onboarding_workflow,
            ...     tag=["user_management", "critical"]
            ... )
            >>> if result.is_ok():
            ...     print(f"Added workflow: {result.value.id}")
        """
        pass

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

        Examples:
            >>> workflows = [
            ...     (WorkflowCard(id="wf1", name="Workflow One"), create_wf1),
            ...     (WorkflowCard(id="wf2", name="Workflow Two"), create_wf2),
            ... ]
            >>> results = mgr.add_workflows(
            ...     workflows,
            ...     tag="production"
            ... )
        """
        pass

    def remove_workflow(self,
                        *,
                        id: Union[str, list[str]] = None,
                        tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                        tag_match_stategy: TagMatchStrategy = TagMatchStrategy.ALL,
                        skip_if_not_exists: bool = False,
                        ) -> Result[Optional[WorkflowCard], Exception] | list[
        Result[Optional[WorkflowCard], Exception]]:
        """
        Remove workflow(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of workflows to remove.
            tag: Single tag or list of tags; removes all workflows with matching tags.
            tag_match_stategy: Strategy for matching tags when using tag parameter.
            skip_if_not_exists: If True, skip non-existent workflows.

        Returns:
            Result[Optional[WorkflowCard], Exception] or list[Result[Optional[WorkflowCard], Exception]]:
                Result object(s) containing the removed workflow card(s) or exception(s).

        Examples:
            >>> # Remove single workflow
            >>> result = mgr.remove_workflow(id="onboarding")
            >>> if result.is_ok() and result.value:
            ...     print(f"Removed workflow: {result.value.id}")
            >>>
            >>> # Remove multiple workflows by tag
            >>> results = mgr.remove_workflow(tag="deprecated")
        """
        pass

    async def get_workflow(self,
                           *,
                           id: Union[str, list[str]] = None,
                           tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                           tag_match_stategy: TagMatchStrategy = TagMatchStrategy.ALL,
                           session: Optional[Session] = None
                           ) -> Optional[Workflow] | list[Optional[Workflow]]:
        """
        Get workflow instance(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of workflows to retrieve.
            tag: Single tag or list of tags; returns all workflows with matching tags.
            tag_match_stategy: Strategy for matching tags when using tag parameter.
            session: Optional session context for the workflows.

        Returns:
            Workflow or list[Workflow]: Workflow instance(s) if found, None otherwise.

        Examples:
            >>> # Get single workflow
            >>> workflow = await mgr.get_workflow(id="onboarding")
            >>>
            >>> # Get multiple workflows
            >>> workflows = await mgr.get_workflow(id=["wf1", "wf2"])
        """
        pass

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

        Examples:
            >>> # Add single tool
            >>> result = mgr.add_tool(calculator_tool, tag="utility")
            >>> if result.is_ok():
            ...     print(f"Added tool: {result.value.id}")
            >>>
            >>> # Add multiple tools
            >>> tools = [tool1, tool2]
            >>> results = mgr.add_tool(
            ...     tools,
            ...     tag=["math", "conversion"]
            ... )
        """
        pass

    def get_tool(self,
                 *,
                 id: Union[str, list[str]] = None,
                 tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                 tag_match_stategy: TagMatchStrategy = TagMatchStrategy.ALL,
                 session: Optional[Session] = None
                 ) -> Optional[Tool] | list[Optional[Tool]]:
        """
        Get tool(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of tools to retrieve.
            tag: Single tag or list of tags; returns all tools with matching tags.
            tag_match_stategy: Strategy for matching tags when using tag parameter.
            session: Optional session context for the tools.

        Returns:
            Tool or list[Tool]: Tool instance(s) if found, None otherwise.

        Examples:
            >>> # Get single tool
            >>> tool = mgr.get_tool(id="calculator")
            >>>
            >>> # Get tools by tag
            >>> tools = mgr.get_tool(tag="api_integration")
        """
        pass

    def remove_tool(self,
                    *,
                    id: Union[str, list[str]] = None,
                    tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                    tag_match_stategy: TagMatchStrategy = TagMatchStrategy.ALL,
                    skip_if_not_exists: bool = False,
                    ) -> Result[Optional[ToolCard], Exception] | list[Result[Optional[ToolCard], Exception]]:
        """
        Remove tool(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of tools to remove.
            tag: Single tag or list of tags; removes all tools with matching tags.
            tag_match_stategy: Strategy for matching tags when using tag parameter.
            skip_if_not_exists: If True, skip non-existent tools.

        Returns:
            Result[Optional[ToolCard], Exception] or list[Result[Optional[ToolCard], Exception]]:
                Result object(s) containing the removed tool card(s) or exception(s).

        Examples:
            >>> # Remove by ID
            >>> result = mgr.remove_tool(id="old_calculator")
            >>> if result.is_ok() and result.value:
            ...     print(f"Removed tool: {result.value.id}")
            >>>
            >>> # Remove by tag
            >>> results = mgr.remove_tool(tag="deprecated_tools")
        """
        pass

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

        Examples:
            >>> result = mgr.add_model(
            ...     id="gpt-4",
            ...     model=create_gpt4_model,
            ...     tag=["llm", "openai", "chat"]
            ... )
            >>> if result.is_ok():
            ...     print(f"Added model: {result.value}")
        """
        pass

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

        Examples:
            >>> models = [
            ...     ("gpt-3.5", create_gpt35_model),
            ...     ("claude-2", create_claude2_model),
            ... ]
            >>> results = mgr.add_models(
            ...     models,
            ...     tag=["llm", "production"]
            ... )
        """
        pass

    def remove_model(self,
                     *,
                     id: Union[str, list[str]] = None,
                     tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                     tag_match_stategy: TagMatchStrategy = TagMatchStrategy.ALL,
                     skip_if_not_exists: bool = False,
                     ) -> Result[str, Exception] | list[Result[str, Exception]]:
        """
        Remove model(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of models to remove.
            tag: Single tag or list of tags; removes all models with matching tags.
            tag_match_stategy: Strategy for matching tags when using tag parameter.
            skip_if_not_exists: If True, skip non-existent models.

        Returns:
            Result[str, Exception] or list[Result[str, Exception]]:
                Result object(s) containing the model ID(s) or exception(s).

        Examples:
            >>> # Remove single model
            >>> result = mgr.remove_model(id="old-model-v1")
            >>> if result.is_ok():
            ...     print(f"Removed model: {result.value}")
            >>>
            >>> # Remove multiple models
            >>> results = mgr.remove_model(id=["model1", "model2"])
        """
        pass

    async def get_model(self,
                        *,
                        id: Union[str, list[str]] = None,
                        tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                        tag_match_stategy: TagMatchStrategy = TagMatchStrategy.ALL,
                        session: Optional[Session] = None) \
            -> Optional[BaseModel] | list[Optional[BaseModel]]:
        """
        Get model instance(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of models to retrieve.
            tag: Single tag or list of tags; returns all models with matching tags.
            tag_match_stategy: Strategy for matching tags when using tag parameter.
            session: Optional session context for the models.

        Returns:
            Model or list[Model]: Model instance(s) if found, None otherwise.

        Examples:
            >>> # Get single model
            >>> model = await mgr.get_model(id="gpt-4")
            >>>
            >>> # Get models by tag
            >>> models = await mgr.get_model(tag="local_models")
        """
        pass

    def add_prompt(self,
                   id: str,
                   template: "PromptTemplate",
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

        Examples:
            >>> result = mgr.add_prompt(
            ...     id="customer_greeting",
            ...     template=PromptTemplate("Hello {{customer_name}}!"),
            ...     tag=["greeting", "customer_service"]
            ... )
            >>> if result.is_ok():
            ...     print(f"Added prompt: {result.value}")
        """
        pass

    def add_prompts(self,
                    prompts: list[Tuple[str, "PromptTemplate"]],
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

        Examples:
            >>> prompts = [
            ...     ("greeting", PromptTemplate("Hello!")),
            ...     ("farewell", PromptTemplate("Goodbye!")),
            ... ]
            >>> results = mgr.add_prompts(prompts, tag="conversation")
        """
        pass

    def remove_prompt(self,
                      *,
                      id: Union[str, list[str]] = None,
                      tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                      tag_match_stategy: TagMatchStrategy = TagMatchStrategy.ALL,
                      skip_if_not_exists: bool = False,
                      ) -> Result[str, Exception] | list[Result[str, Exception]]:
        """
        Remove prompt template(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of prompts to remove.
            tag: Single tag or list of tags; removes all prompts with matching tags.
            tag_match_stategy: Strategy for matching tags when using tag parameter.
            skip_if_not_exists: If True, skip non-existent prompts.

        Returns:
            Result[str, Exception] or list[Result[str, Exception]]:
                Result object(s) containing the prompt ID(s) or exception(s).

        Examples:
            >>> # Remove single prompt
            >>> result = mgr.remove_prompt(id="old_greeting")
            >>> if result.is_ok():
            ...     print(f"Removed prompt: {result.value}")
            >>>
            >>> # Remove by tag
            >>> results = mgr.remove_prompt(tag="deprecated")
        """
        pass

    def get_prompt(self,
                   *,
                   id: Union[str, list[str]] = None,
                   tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                   tag_match_stategy: TagMatchStrategy = TagMatchStrategy.ALL,
                   ) -> Optional["PromptTemplate"] | list[Optional["PromptTemplate"]]:
        """
        Get prompt template(s) by ID or tag.

        Args:
            id: Single ID or list of IDs of prompts to retrieve.
            tag: Single tag or list of tags; returns all prompts with matching tags.
            tag_match_stategy: Strategy for matching tags when using tag parameter.

        Returns:
            PromptTemplate or list[PromptTemplate]: Prompt template instance(s) if found, None otherwise.

        Examples:
            >>> # Get single prompt
            >>> prompt = mgr.get_prompt(id="customer_greeting")
            >>>
            >>> # Get multiple prompts
            >>> prompts = mgr.get_prompt(id=["greeting", "farewell"])
        """
        pass

    async def get_tool_info(self,
                            *,
                            id: Union[str, list[str]] = None,
                            type: Union[str, list[str]] = None,
                            tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                            tag_match_stategy: TagMatchStrategy = TagMatchStrategy.ALL,
                            ignore_exception: bool = False,
                            ) -> Optional[ToolInfo] | list[Optional[ToolInfo]]:
        """
        Get tool information/metadata by ID, type, or tag.

        Args:
            id: Single ID or list of IDs of tools to get info for.
            type: Single type or list of types to filter tools by.
                Common types: ["function", "mcp", "workflow"].
            tag: Single tag or list of tags; returns info for all tools with matching tags.
            tag_match_stategy: Strategy for matching tags when using tag parameter.
            ignore_exception: If True, ignore exceptions and return None for failing items.

        Returns:
            ToolInfo or list[ToolInfo]: Tool information instance(s) if found, None otherwise.

        Examples:
            >>> # Get info by ID
            >>> info = await mgr.get_tool_info(id="calculator")
            >>>
            >>> # Get info by type
            >>> infos = await mgr.get_tool_info(type="workflow")
            >>>
            >>> # Get info by multiple criteria
            >>> infos = await mgr.get_tool_info(
            ...     tag=["api", "production"],
            ...     type="function"
            ... )
        """
        pass

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

        Examples:
            >>> config = McpServerConfig(
            ...     name="github",
            ...     command="npx",
            ...     args=["@modelcontextprotocol/server-github"]
            ... )
            >>> result = await mgr.add_mcp_server(
            ...     server_config=config,
            ...     tag="version_control",
            ...     expiry_time=time.time() + 86400  # 1 day from now
            ... )
            >>> if result.is_ok():
            ...     print(f"Added MCP server: {result.value}")
        """
        pass

    async def refresh_mcp_server(self,
                                 server_name: Union[str, list[str]],
                                 *,
                                 tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                                 tag_match_stategy: TagMatchStrategy = TagMatchStrategy.ALL,
                                 ignore_exception: bool = False,
                                 skip_if_not_exists: bool = False,
                                 ) -> Result[str, Exception] | list[Result[str, Exception]]:
        """
        Refresh MCP server toolinfos(s) by name.

        Args:
            server_name: Single or list of MCP server names to refresh.
            tag: Optional tag to filter servers to refresh.
            tag_match_stategy: Strategy for matching tags when using tag parameter.
            ignore_exception: If True, continue refreshing other servers if one fails.
            skip_if_not_exists: If True, skip non-existent servers.

        Returns:
            Result[str, Exception] or list[Result[str, Exception]]:
                Result object(s) containing the server name(s) or exception(s).

        Examples:
            >>> # Refresh single server
            >>> result = await mgr.refresh_mcp_server("github")
            >>> if result.is_ok():
            ...     print(f"Refreshed MCP server: {result.value}")
            >>>
            >>> # Refresh multiple servers
            >>> results = await mgr.refresh_mcp_server(
            ...     ["github", "filesystem"],
            ...     ignore_exception=True
            ... )
        """
        pass

    async def remove_mcp_server(self,
                                *,
                                server_name: Optional[Union[str, list[str]]] = None,
                                tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                                tag_match_stategy: TagMatchStrategy = TagMatchStrategy.ALL,
                                skip_if_not_exists: bool = False,
                                ignore_exception: bool = False,
                                ) -> Result[str, Exception] | list[Result[str, Exception]]:
        """
        Remove MCP server(s) by name or tag.

        Args:
            server_name: Single or list of MCP server names to remove.
            tag: Single tag or list of tags; removes all servers with matching tags.
            tag_match_stategy: Strategy for matching tags when using tag parameter.
            skip_if_not_exists: If True, skip non-existent servers.
            ignore_exception: If True, continue removing other servers if one fails.

        Returns:
            Result[str, Exception] or list[Result[str, Exception]]:
                Result object(s) containing the server name(s) or exception(s).

        Examples:
            >>> # Remove by name
            >>> result = await mgr.remove_mcp_server(server_name="old_server")
            >>> if result.is_ok():
            ...     print(f"Removed MCP server: {result.value}")
            >>>
            >>> # Remove by tag
            >>> results = await mgr.remove_mcp_server(tag="deprecated")
        """
        pass

    async def get_mcp_tool(self,
                           *,
                           name: Union[str, list[str]],
                           server_name: Union[str, list[str]],
                           tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                           tag_match_stategy: TagMatchStrategy = TagMatchStrategy.ALL,
                           session: Optional[Session] = None
                           ) -> Optional[Tool] | list[Optional[Tool]]:
        """
        Get MCP tool(s) by name and server.

        Args:
            name: Single or list of MCP tool names to retrieve.
            server_name: Single or list of MCP server names containing the tools.
            tag: Optional tag to filter servers/tools.
            tag_match_stategy: Strategy for matching tags when using tag parameter.
            session: Optional session context for the tools.

        Returns:
            Tool or list[Tool]: MCP tool instance(s) if found, None otherwise.

        Examples:
            >>> # Get single tool
            >>> tool = await mgr.get_mcp_tool(
            ...     name="search_code",
            ...     server_name="github"
            ... )
            >>>
            >>> # Get multiple tools
            >>> tools = await mgr.get_mcp_tool(
            ...     name=["read_file", "write_file"],
            ...     server_name="filesystem"
            ... )
        """
        pass

    async def get_mcp_tool_info(self,
                                *,
                                name: Union[str, list[str]] = None,
                                server_name: Union[str, list[str]] = None,
                                tag: Optional[Union[Tag, list[Tag]]] = GLOBAL,
                                tag_match_stategy: TagMatchStrategy = TagMatchStrategy.ALL,
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
            tag_match_stategy: Strategy for matching tags when using tag parameter.
            ignore_exception: If True, ignore exceptions for individual items.

        Returns:
            ToolInfo or list[ToolInfo]: MCP tool information instance(s) if found, None otherwise.

        Examples:
            >>> # Get info for single tool
            >>> info = await mgr.get_mcp_tool_info(
            ...     name="search_code",
            ...     server_name="github"
            ... )
            >>>
            >>> # Get info for all tools in a server
            >>> infos = await mgr.get_mcp_tool_info(server_name="filesystem")
        """
        pass

    def get_resource_by_tag(self,
                            tag: Tag) -> Optional[list["BaseCard"]]:
        """
        Retrieve all resources associated with a specific tag.

        Args:
            tag: Tag to search for.

        Returns:
            List of BaseCard instances representing resources with the specified tag,
            or None if no resources found.

        Examples:
            >>> resources = mgr.get_resource_by_tag("customer_service")
            >>> if resources:
            ...     for card in resources:
            ...         print(f"Resource: {card.id}, Type: {type(card).__name__}")
        """
        pass

    def list_tags(self) -> list[Tag]:
        """
        List all tags currently in use across all resources.

        Returns:
            List of unique tag strings.

        Examples:
            >>> tags = mgr.list_tags()
            >>> print(f"Available tags: {tags}")
        """
        pass

    def has_tag(self, tag: str) -> bool:
        """
            Check if the specified tag exists in the resource_mgr.
        """
        pass

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

        Examples:
            >>> # Remove single tag
            >>> result = mgr.remove_tag("deprecated")
            >>> if result.is_ok():
            ...     print(f"Removed tag: {result.value}")
            >>>
            >>> # Remove multiple tags
            >>> results = mgr.remove_tag(["temp", "test"])
        """
        pass

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

        Examples:
            >>> result = mgr.update_resource_tag(
            ...     id="translator_agent",
            ...     tag=["language", "production", "priority_high"]
            ... )
            >>> if result.is_ok():
            ...     print(f"Updated tags: {result.value}")
        """
        pass

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

        Examples:
            >>> result = mgr.add_resource_tag(
            ...     id="calculator_tool",
            ...     tag="math"
            ... )
            >>> if result.is_ok():
            ...     print(f"Current tags: {result.value}")
        """
        pass

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

        Examples:
            >>> result = mgr.remove_resource_tag(
            ...     id="customer_service_agent",
            ...     tag="deprecated"
            ... )
            >>> if result.is_ok():
            ...     print(f"Remaining tags: {result.value}")
        """
        pass

    def get_resource_tag(self, id: str) -> Optional[list[Tag]]:
        """
        Get all tags associated with a resource.

        Args:
            id: Resource identifier.

        Returns:
            List of tags associated with the resource, or None if resource not found.

        Examples:
            >>> tags = mgr.get_resource_tag("onboarding_workflow")
            >>> if tags:
            ...     print(f"Workflow tags: {tags}")
        """
        pass

    def resource_has_tag(self, id: str, tag: str) -> bool:
        """
        Check if a specific resource is associated with the given tag.
        """
        pass