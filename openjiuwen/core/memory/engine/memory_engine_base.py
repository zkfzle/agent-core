from abc import ABC, abstractmethod
from typing import Any
from sqlalchemy import Engine
from openjiuwen.core.memory.messages.messages import SeqMessage
from openjiuwen.core.memory.config.config import Config
from openjiuwen.core.memory.store.base_kv_store import BaseKVStore
from openjiuwen.core.memory.store.base_vector_store import BaseVectorStore
from openjiuwen.core.utils.llm.base import BaseChatModel


class MemoryEngineBase(ABC):
    def __init__(self, config: Config, llm_base: BaseChatModel = None):
        pass
    
    @abstractmethod
    def init_mem_store(
        self,
        vector_db_instance: BaseVectorStore,
        db_engine_instance: Engine,
        kv_db_instance: BaseKVStore
    ):
        pass
    
    @abstractmethod
    def set_app_config(self, app_id: str, config_key: str, config_value: Any):
        pass

    @abstractmethod
    def add_conversation_message(
        self,
        user_id: str,
        app_id: str,
        messages: SeqMessage,
        request_config: dict[str, Any] = None,
        session_id: str = None,
        llm: BaseChatModel = None
    ) -> str:
        pass
    
    @abstractmethod
    async def aadd_conversation_message(
        self,
        user_id: str,
        app_id: str,
        messages: SeqMessage,
        request_config: dict[str, Any] = None,
        session_id: str = None,
        llm: BaseChatModel = None
    ) -> str:
        pass
    
    @abstractmethod
    def get_recent_message(self, user_id: str, app_id: str, session_id: str = None) -> list[SeqMessage]:
        pass
    
    @abstractmethod
    def get_message_by_id(self, msg_id: str) -> SeqMessage:
        pass
    
    @abstractmethod
    def delete_mem_by_id(self, mem_id: str) -> bool:
        pass
    
    @abstractmethod
    def delete_mem_by_user_id(self, user_id: str, app_id: str) -> bool:
        pass
    
    @abstractmethod
    def delete_user_profile_by_user_id(self, user_id: str, app_id: str) -> bool:
        pass
    
    @abstractmethod
    def update_mem_by_id(self, mem_id: str, memory: str) -> bool:
        pass
    
    @abstractmethod
    def get_user_variable(self, user_id: str, app_id: str, name: str) -> str:
        pass
    
    @abstractmethod
    def list_user_variables(self, user_id: str, app_id: str) -> dict[str, str]:
        pass
    
    @abstractmethod
    def search_user_mem(self, user_id: str, app_id: str, query: str, num: int) -> list[dict[str, Any]]:
        pass
    
    @abstractmethod
    def list_user_mem(self, user_id: str, app_id: str, num: int, page: int) -> list[dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_user_profile_by_topics(self, user_id: str, app_id: str, topics: list[str]) -> dict[str, str]:
        pass