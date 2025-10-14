from typing import List, Optional, Tuple

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.tracer.decorator import decrate_model_with_trace
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.runtime.thread_safe_dict import ThreadSafeDict


class ModelMgr:
    """
    Thread-Safe Model Manager
    """
    __slots__ = ("_models",)

    def __init__(self) -> None:
        self._models: ThreadSafeDict[str, BaseChatModel] = ThreadSafeDict()

    def add_model(self, model_id: str, model: BaseChatModel) -> None:
        if model_id is None:
            raise JiuWenBaseException(StatusCode.RUNTIME_MODEL_ADD_FAILED.code,
                                      StatusCode.RUNTIME_MODEL_ADD_FAILED.errmsg.format(
                                          reason="model_id is invalid, can not be None"))
        if model is None:
            raise JiuWenBaseException(StatusCode.RUNTIME_MODEL_ADD_FAILED.code,
                                      StatusCode.RUNTIME_MODEL_ADD_FAILED.errmsg.format(
                                          reason="model is invalid, can not be None"))
        self._models[model_id] = model

    def add_models(self, models: List[Tuple[str, BaseChatModel]]) -> None:
        for model_id, model in models:
            self.add_model(model_id, model)

    def remove_model(self, model_id: str) -> Optional[BaseChatModel]:
        if model_id is None:
            return None
        return self._models.pop(model_id, None)

    def get_model(self, model_id: str, runtime=None) -> Optional[BaseChatModel]:
        if model_id is None:
            raise JiuWenBaseException(StatusCode.RUNTIME_MODEL_GET_FAILED.code,
                                      StatusCode.RUNTIME_MODEL_GET_FAILED.errmsg.format(
                                          reason="model_id is invalid, can not be None"))
        model = self._models.get(model_id)
        if not model or not runtime or not runtime.tracer():
            return model
        return decrate_model_with_trace(WrappedBaseChatModel(model), runtime)


class WrappedBaseChatModel(BaseChatModel):
    def __init__(self, model: BaseChatModel):
        """
        初始化模型，子类应该在这里设置自己的配置参数
        """
        super().__init__(model.api_key, model.api_base, model.max_retrie, model.max_retrie)
        self.inner = model

    def _invoke(self, model_name: str, messages, tools=None, temperature: float = 0.1,
                top_p: float = 0.1, **kwargs):
        return self.inner._invoke(model_name, messages, tools=tools, temperature=temperature, top_p=top_p, **kwargs)

    async def _ainvoke(self, model_name, messages, tools=None, temperature: float = 0.1,
                       top_p: float = 0.1, **kwargs):
        return await self.inner.ainvoke(model_name, messages, tools=tools, temperature=temperature, top_p=top_p,
                                        **kwargs)

    def _stream(self, model_name: str, messages, tools=None, temperature: float = 0.1,
                top_p: float = 0.1, **kwargs):
        yield from self.inner._stream(model_name, messages, tools=tools, temperature=temperature, top_p=top_p,
                                      **kwargs)

    async def _astream(self, model_name: str, messages, tools=None, temperature: float = 0.1,
                       top_p: float = 0.1, **kwargs):
        result = self.inner._astream(model_name, messages, tools=tools, temperature=temperature, top_p=top_p, **kwargs)
        async for item in result:
            yield item

    def model_provider(self):
        return self.inner.model_provider()
