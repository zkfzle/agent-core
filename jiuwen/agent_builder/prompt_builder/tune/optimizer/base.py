#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from abc import abstractmethod
from typing import Dict, Optional, Any, List, Callable
import copy

from pydantic import BaseModel, Field

from jiuwen.agent_builder.prompt_builder.tune.base import EvaluatedCase
from jiuwen.core.agent.agent import Agent
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.common.logging import logger
from jiuwen.core.utils.llm.messages import BaseMessage, ToolInfo
from jiuwen.core.utils.llm_call.base import LLMCall
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.agent_builder.prompt_builder.tune.utils import TuneUtils
from jiuwen.agent_builder.prompt_builder.tune.dataset.case_loader import CaseLoader
from jiuwen.agent_builder.prompt_builder.tune.evaluator.evaluator import BaseEvaluator


class BaseOptimizer:
    def __init__(self,
                 agent: Agent,
                 **kwargs
                 ):
        from jiuwen.agent.chat_agent import ChatAgent
        if not isinstance(agent, ChatAgent):
            raise TypeError("optimizer only support Chat Agent right now.")
        self._agent = agent
        self._parameters: Dict[str, TextualParameter] = {}
        for name, llm_call in agent.get_llm_calls().items():
            self._parameters[name] = TextualParameter(llm_call)
        self._history = OptimizeHistory()
        self._bad_cases: List[EvaluatedCase] = []

    def backward(self,
                 evaluated_cases: List[EvaluatedCase],
                 ):
        self._batch_set_optimizer_callback(self.trace_callback)
        self._get_bad_cases(evaluated_cases)
        try:
            self._backward(evaluated_cases)
        except Exception as e:
            self._batch_set_optimizer_callback(None)
            raise JiuWenBaseException(
                StatusCode.AGENT_BUILDER_AGENT_OPTIMIZER_BACKWORD_ERROR.code,
                StatusCode.AGENT_BUILDER_AGENT_OPTIMIZER_BACKWORD_ERROR.errmsg.format(
                    error_msg=f"{str(e)}"
                )
            )
        self._batch_set_optimizer_callback(None)

    def update(self) -> Optional[Agent]:
        try:
            optimized_agent = self._update()
            for name, llm_call in optimized_agent.get_llm_calls().items():
                logger.info(f"[llm_call name]: {name}\n"
                            f"[optimized prompt]:{str(llm_call.get_system_prompt().content)}")
            self._history.clear_history()
            return optimized_agent
        except Exception as e:
            self._history.clear_history()
            raise JiuWenBaseException(
                StatusCode.AGENT_BUILDER_AGENT_OPTIMIZER_UPDATE_ERROR.code,
                StatusCode.AGENT_BUILDER_AGENT_OPTIMIZER_UPDATE_ERROR.errmsg.format(
                    error_msg=f"{str(e)}"
                )
            )

    @abstractmethod
    def _update(self) -> Optional[Agent]:
        return self._agent

    @abstractmethod
    def _backward(self,
                 evaluated_cases: List[EvaluatedCase] ,
                 ) -> Optional[Agent]:
        pass

    def parameters(self):
        return self._parameters

    def prepare(self,
                case_loader: CaseLoader,
                evaluator: Optional[BaseEvaluator] = None,
                ) -> Optional[Agent]:
        return self._agent

    def trace_callback(self,
                       input: Dict[str, str],
                       output: BaseMessage,
                       runtime: Runtime
                       ):
        trace_node = TraceNode(
            case_id=runtime.session_id(),
            # TODO: get llm call id from runtime
            llm_call_id="llm_call",
            inputs=input,
            outputs=TuneUtils.get_output_string_from_message(output)
        )
        self._history.add_history(runtime.session_id(), trace_node)

    def _batch_set_optimizer_callback(self, callback: Optional[Callable]) -> None:
        for _, param in self._parameters.items():
            param.llm_call.set_optimizer_callback(callback)

    def _get_bad_cases(self, evaluated_cases: List[EvaluatedCase]) -> List[EvaluatedCase]:
        bad_cases = [case for case in evaluated_cases if case.score == 0]
        self._bad_cases = bad_cases
        return bad_cases


class TextualParameter:
    def __init__(self, llm_call: LLMCall):
        self.llm_call = copy.deepcopy(llm_call)
        self.gradients: Dict[str, str] = {}

    def set_gradient(self, name: str, gradient: str):
        self.gradients[name] = gradient

    def get_gradient(self, name: str) -> Optional[str]:
        return self.gradients.get(name)


class TraceNode(BaseModel):
    case_id: str = Field(...)
    llm_call_id: str = Field(...)
    inputs: Dict[str, Any] = Field(...)
    outputs: str = Field(default="")
    history: List[BaseMessage] = Field(default=[])
    tools: List[ToolInfo] | List[Dict] = Field(default=[])


class OptimizeHistory:
    def __init__(self):
        self._trajectory: Dict[str, List[TraceNode]] = {}

    def add_history(self, case_id: str, node: TraceNode):
        if case_id not in self._trajectory:
            self._trajectory[case_id] = []
        self._trajectory[case_id].append(node)

    def get_history(self, case_id: str) -> Optional[List[TraceNode]]:
        if case_id not in self._trajectory:
            return None
        return self._trajectory[case_id]

    def clear_history(self):
        self._trajectory = {}