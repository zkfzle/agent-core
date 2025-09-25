from typing import List, Optional, Tuple

from jiuwen.core.tracer.decorator import decrate_model_with_trace
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.context.controller_context.thread_safe_dict import ThreadSafeDict


class ModelMgr:
    """线程安全单机模型管理器"""
    __slots__ = ("_models",)

    def __init__(self) -> None:
        self._models: ThreadSafeDict[str, BaseChatModel] = ThreadSafeDict()

    def add_model(self, model_id: str, model: BaseChatModel) -> None:
        self._models[model_id] = model

    def add_models(self, models: List[Tuple[str, BaseChatModel]]) -> None:
        self._models.update(models)

    def remove_model(self, model_id: str) -> bool:
        return self._models.pop(model_id, None) is not None

    def get_model(self, model_id: str, runtime=None) -> Optional[BaseChatModel]:
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
