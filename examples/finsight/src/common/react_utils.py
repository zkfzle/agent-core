from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from pydantic import BaseModel

from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.utils.llm.messages import AIMessage, BaseMessage, ToolMessage
from openjiuwen.core.common.logging import logger


class Action(BaseModel):
    action: str
    content: Any
    id: Any = None


class ActionResult(BaseModel):
    message: BaseMessage
    is_finished: bool = False


class ActionSpace(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def parse_llm_response(self, message: AIMessage) -> List[Action]:
        pass

    @abstractmethod
    async def execute(self, action: Action) -> ActionResult:
        pass


class FunctionCallSpace(ActionSpace):
    def __init__(self):
        pass

    def parse_llm_response(self, message: AIMessage) -> List[Action]:
        return [
            Action(
                action=tool_call.name,
                content=tool_call.arguments,
                id=tool_call.id
            ) for tool_call in message.tool_calls
        ]

    async def execute(self, action: Action) -> ActionResult:
        result = await Runner.run_tool(action.action, action.content)
        return ActionResult(message=ToolMessage(content=str(result), tool_call_id=action.id))


async def async_run_react_loop(llm, messages: List[BaseMessage], max_iterations: int, action_space: ActionSpace = FunctionCallSpace(), *args, **kwargs) -> Tuple[List[BaseMessage], bool]:
    stop = False
    curr_iteration = 0
    while curr_iteration < max_iterations and not stop:
        curr_iteration += 1

        response = await llm.ainvoke(messages, *args, **kwargs)
        logger.info(f"Current iteration: {curr_iteration}, response: {response.content}")
        messages.append(response)

        actions = action_space.parse_llm_response(response)
        if not actions:
            return messages, False

        for action in actions:
            action_result = await action_space.execute(action)
            messages.append(action_result.message)
            stop = stop or action_result.is_finished

    return messages, curr_iteration >= max_iterations

