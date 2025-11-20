import asyncio
from typing import Any
from sqlalchemy.engine import Engine

from openjiuwen.core.memory.manage.write_manager import WriteManager
from openjiuwen.core.memory.manage.data_id_manager import DataIdManager
from openjiuwen.core.memory.manage.variable_manager import VariableManager
from openjiuwen.core.memory.mem_unit.memory_unit import BaseMemoryUnit, UserProfileUnit, VariableUnit, MemoryType
from openjiuwen.core.memory.manage.user_profile_manager import UserProfileManager
from openjiuwen.core.memory.messages.messages import SeqMessage
from openjiuwen.core.memory.search.search_manager.search_manager import SearchManager
from openjiuwen.core.memory.store.base_kv_store import BaseKVStore
from openjiuwen.core.memory.engine.memory_engine_base import MemoryEngineBase
from openjiuwen.core.memory.config.config import Config
from openjiuwen.core.memory.config.config_manager import ConfigManger
from openjiuwen.core.memory.generation.generation import Generator
from openjiuwen.core.memory.memory_logging import get_logger
from openjiuwen.core.memory.store.sql_db_store import SqlDbStore
from openjiuwen.memory.store.user_mem_store import UserMemStore
from openjiuwen.core.memory.store.base_semantic_store import BaseSemanticStore
from openjiuwen.core.memory.manage.message_manager import MessageManager
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.llm.messages import BaseMessage

logger = get_logger()


def _check_user_and_app_id(user_id: str, app_id: str, context="Operation"):
    if not user_id or user_id.strip() == "":
        raise ValueError(f"{context} failed: user_id is empty.")
    if not app_id or app_id.strip() == "":
        raise ValueError(f"{context} failed: app_id is empty.")
    
class MemoryEngine(MemoryEngineBase):
    def __init__(self, config: Config, llm_base: BaseModelClient = None):
        self.config_manager = ConfigManger(config)
        self.llm_base = llm_base
        self.user_profile_manager: UserProfileManager = None
        self.write_manager: WriteManager = None
        self.variable_manager: VariableManager = None
        self.message_manager: MessageManager = None
        self.generator: Generator = None
        self.search_manager: SearchManager = None
    
    def init_mem_store(self,
                       semantic_db_instance: BaseSemanticStore,
                       db_engine_instance: Engine,
                       kv_db_instance: BaseKVStore):
        data_id_generator = DataIdManager(kv_db_instance)
        user_mem_store = UserMemStore(kv_db_instance)
        self.user_profile_manager = UserProfileManager(
            semantic_db_instance,
            user_mem_store,
            data_id_generator
        )
        self.variable_manager = VariableManager(kv_db_instance)
        sql_db_store = SqlDbStore(db_engine_instance)
        self.message_manager = MessageManager(sql_db_store, data_id_generator)
        managers = {
            MemoryType.USER_PROFILE.value: self.user_profile_manager,
            MemoryType.VARIABLE.value: self.variable_manager
        }
        self.search_manager = SearchManager(managers, user_mem_store)
        self.generator = Generator(self.search_manager)
        self.write_manager = WriteManager(managers, user_mem_store)
        
    def set_app_config(self, app_id: str, config_key: str, config_value: Any):
        if app_id is None or app_id.strip() == "":
            logger.warning("set_app_config failed: app_id is empty.")
            return False
        if config_key is None or config_key.strip() == "":
            logger.warning("set_app_config failed: config_key is empty.")
            return False
        if config_value is None:
            logger.warning("set_app_config failed: config_value is None.")
            return False
        self.config_manager.set_app_config(app_id, config_key, config_value)
        return True
    
    def add_conversation_message(
        self,
        user_id: str,
        app_id: str,
        messages: SeqMessage,
        request_config: dict[str, Any] = None,
        session_id: str = None,
        llm: BaseModelClient = None
    ) -> str:
        llm = llm if llm else self.llm_base
        if not self.message_manager:
            raise ValueError("Message Manager is not initialized. Please call init_mem_store first.")
        current_message = BaseMessage(role=messages.role, content=messages.content)
        config = self.config_manager.get_config(app_id, request_config)
        threshold = config.realtime_process_config.window_size
        user_profile_custom_define = config.realtime_process_config.user_profile_custom_define
        history_messages = self.message_manager.get(
            user_id=user_id,
            app_id=app_id,
            session_id=session_id,
            message_len=threshold
        )
        message_mem_id = self.message_manager.add(
            user_id=user_id,
            app_id=app_id,
            role=messages.role,
            content=messages.content,
        )
        all_memory: list[BaseMemoryUnit] = self.generator.gen_all_memory(
            app_id=app_id,
            user_id=user_id,
            messages=[current_message],
            history_messages=history_messages,
            session_id=session_id,
            config=config,
            base_chat_model=llm,
            message_mem_id=message_mem_id,
            user_define=user_profile_custom_define
        )
        self.write_manager.add_mem(all_memory)
        return message_mem_id
    
    async def aadd_conversation_message(
        self,
        user_id: str,
        app_id: str,
        messages: SeqMessage,
        request_config: dict[str, Any] = None,
        session_id: str = None,
        llm: BaseModelClient = None
    ) -> str:
        loop = asyncio.get_event_loop()
        message_mem_id = await loop.run_in_executor(None, self.add_conversation_message,
                                                   user_id, app_id, messages,
                                                   request_config, session_id, llm)
        return message_mem_id
    
    def get_recent_message(self, user_id: str, app_id: str, session_id: str = None) -> list[SeqMessage]:
        if not self.message_manager:
            raise ValueError("Message Manager is not initialized. Please call init_mem_store first.")
        return self.message_manager.get(
            user_id=user_id,
            app_id=app_id,
            session_id=session_id
        )
    
    def get_message_by_id(self, msg_id: str) -> SeqMessage:
        if not self.message_manager:
            raise ValueError("Message Manager is not initialized. Please call init_mem_store first.")
        return self.message_manager.get_by_id(msg_id)[0]
    
    def delete_mem_by_id(self, mem_id: str) -> bool:
        if not self.write_manager:
            raise ValueError("Write Manager is not initialized. Please call init_mem_store first.")
        self.write_manager.delete_mem_by_id(mem_id)
        return True
    
    def delete_mem_by_user_id(self, user_id: str, app_id: str) -> bool:
        if not self.write_manager:
            raise ValueError("Write Manager is not initialized. Please call init_mem_store first.")
        self.write_manager.delete_mem_by_user_id(user_id=user_id, app_id=app_id)
        return True
    
    def delete_user_profile_by_user_id(self, user_id: str, app_id: str) -> bool:
        if not self.write_manager:
            raise ValueError("Write Manager is not initialized. Please call init_mem_store first.")
        self.user_profile_manager.delete_by_user_id(user_id=user_id, app_id=app_id)
        return True
    
    def update_mem_by_id(self, mem_id: str, memory: str) -> bool:
        if not self.write_manager:
            raise ValueError("Write Manager is not initialized. Please call init_mem_store first.")
        self.write_manager.update_mem_by_id(mem_id, memory)
        return True
    
    def get_user_variable(self, user_id: str, app_id: str, name: str) -> str:
        _check_user_and_app_id(user_id, app_id, "Query Variable")
        if not name or name.strip() == "":
            raise ValueError("Query Variable failed: variable name is empty.")
        return self.search_manager.get_user_variable(user_id, app_id, name)
    
    def search_user_mem(self, user_id: str, app_id: str, query: str, num: int) -> list[dict[str, Any]]:
        _check_user_and_app_id(user_id, app_id, "Search User Memory")
        if not query or query.strip() == "":
            raise ValueError("Search User Memory failed: query is empty.")
        if num is None or num <= 0:
            raise ValueError("Search User Memory failed: num must be greater than 0.")
        return self.search_manager.search(query=query, app_id=app_id, top_k=num, user_id=user_id)
    
    def list_user_variables(self, user_id: str, app_id: str) -> dict[str, str]:
        _check_user_and_app_id(user_id, app_id, "List User Variables")
        return self.search_manager.get_all_user_variables(user_id, app_id)
    
    def list_user_mem(self, user_id: str, app_id: str, num: int, page: int) -> list[dict[str, Any]]:
        if num is None or num <= 0:
            raise ValueError("List User Memory failed: num must be greater than 0.")
        if page is None or page <= 0:
            raise ValueError("List User Memory failed: page must be greater than 0.")
        _check_user_and_app_id(user_id, app_id, "List User Memory")
        return self.search_manager.list_user_mem(user_id=user_id, app_id=app_id, nums=num, pages=page)
    
    def get_user_profile_by_topics(self, user_id: str, app_id: str, topics: list[str]) -> dict[str, str]:
        logger.info("get_user_profile_by_topics is not implemented yet.")
        return {}