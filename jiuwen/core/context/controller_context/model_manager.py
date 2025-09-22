from typing import List, Optional, Tuple
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

    def get_model(self, model_id: str) -> Optional[BaseChatModel]:
        return self._models.get(model_id)

