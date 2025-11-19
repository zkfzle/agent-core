from .memory_info import ExtractedData
from openjiuwen.core.memory.generation.variable_extractor import ComprehensionExtractor
from ..search.search_manager.search_manager import SearchManager
from ..mem_unit.memory_unit import MemoryType, BaseMemoryUnit, VariableUnit, UserProfileUnit
from openjiuwen.core.memory.generation.categorizer import Categorizer
from openjiuwen.core.memory.generation.user_profile_extractor import UserProfileExtractor
from openjiuwen.core.memory.generation.conflict_resolution import ConflictResolution
from openjiuwen.core.common.logging import logger
from ..config.config import Config
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.llm.messages import BaseMessage

category_to_class = {
    "user_profile": MemoryType.USER_PROFILE
}


def _generate_extract(
    config: Config,
    history_messages: list[BaseMessage],
    messages: list[BaseMessage],
    base_chat_model: BaseModelClient
) -> list[ExtractedData]:
    history_summary = ""
    for msg in history_messages:
        history_summary += f"{msg.role}: {msg.content}\n"
    return ComprehensionExtractor.extract(
        messages,
        BaseMessage(content=history_summary, role=""),
        base_chat_model,
        config
    )


def _generate_user_profile(
    config: Config,
    history_messages: list[BaseMessage],
    messages: list[BaseMessage],
    base_chat_model: BaseModelClient,
    user_define: dict[str, str] = None
) -> dict[str, str]:
    return UserProfileExtractor.GetUserProfile(
        messages,
        history_messages,
        base_chat_model,
        config,
        user_define
    )


def _get_conflict_input(
    user_id: str,
    app_id: str,
    new_message: str,
    search_manager: SearchManager
):
    historical_profiles = []
    search_results = search_manager.search(
        new_message,
        top_k=5,
        search_type=MemoryType.USER_PROFILE.value,
        user_id=user_id,
        app_id=app_id
    )
    for search_result in search_results:
        historical_profiles.append((
            search_result['id'],
            search_result['mem'],
            search_result['score']
        ))
    input_memories_map: dict[int, str] = {}
    input_memory_ids_map: dict[int, str] = {}
    input_memories: list[str] = []
    i = 1
    for historical in historical_profiles:
        mem_id, mem_content, _ = historical
        input_memories.append(mem_content)
        input_memories_map[i] = mem_content
        input_memory_ids_map[i] = mem_id
        i += 1
    return input_memories, input_memories_map, input_memory_ids_map


def _process_conflict_info(conflict_info: list[dict], input_memory_ids_map: dict[int, str]) -> list[dict]:
    process_conflict_info = []
    for conflict in conflict_info:
        conf_id = int(conflict['id'])
        conf_mem = conflict['text']
        conf_event = conflict['event']
        if conf_id == 0:
            process_conflict_info.append({
                "id": '-1',
                "text": conf_mem,
                "event": conf_event
            })
            continue
        map_id = input_memory_ids_map[conf_id]
        process_conflict_info.append({
            "id": map_id,
            "text": conf_mem,
            "event": conf_event
        })
    return process_conflict_info


class Generator:
    def __init__(self, search_manager: SearchManager) -> None:
        self._search_manager = search_manager
        
    def gen_all_memory(self, **kwargs) -> list[BaseMemoryUnit]:
        """Generate all memory units based on input"""
        messages = kwargs.get("messages")
        config = kwargs.get("config")
        model = kwargs.get("base_chat_model")
        user_define = kwargs.get("user_define", None)
        user_id = kwargs.get("user_id")
        app_id = kwargs.get("app_id")
        session_id = kwargs.get("session_id")
        history_messages = kwargs.get("history_messages")
        message_mem_id = kwargs.get("message_mem_id")
        if not all([messages, config, user_id, app_id, session_id]) :
            logger.error("messages, config, user_id, app_id, session_id are required parameters")
        if not model:
            logger.error("base_chat_model is required parameter")
        categorizer = Categorizer()
        all_memory_results = []
        variable_units = self.gen_extracted_data(
            messages=messages,
            user_id=user_id,
            app_id=app_id,
            history_messages=history_messages,
            config=config,
            base_chat_model=model
        )
        all_memory_results += variable_units
        categories = categorizer.GetCategories(
            messages,
            history_messages,
            model,
            config
        )
        merged_units = self._categories_to_memory_unit(
            categories=categories,
            history_messages=history_messages,
            messages=messages,
            user_id=user_id,
            app_id=app_id,
            config=config,
            base_chat_model=model,
            message_mem_id=message_mem_id,
            user_define=user_define
        )
        all_memory_results += merged_units
        return all_memory_results
    
    def gen_extracted_data(
        self,
        user_id: str,
        app_id: str,
        messages: list[BaseMessage],
        history_messages: list[BaseMessage],
        config: Config,
        base_chat_model: BaseModelClient
    ) -> list[VariableUnit]:
        """Generate extracted variable memory units based on input"""
        extracted_data = _generate_extract(
            config,
            history_messages,
            messages,
            base_chat_model
        )
        variable_units = []
        for tmp_data in extracted_data:
            variable_units.append(VariableUnit(
                    user_id=user_id,
                    app_id=app_id,
                    mem_type=MemoryType.VARIABLE,
                    variable_name=tmp_data.key,
                    variable_mem=tmp_data.value
                ))
        return variable_units
        
    def gen_user_profile(
        self,
        user_id: str,
        app_id: str,
        messages: list[BaseMessage],
        history_messages: list[BaseMessage],
        config: Config,
        base_chat_model: BaseModelClient,
        message_mem_id: str,
        user_define: dict[str, str] = None
    ) -> list[UserProfileUnit]:
        """Generate user profile memory unit based on input"""
        user_profile_memory = _generate_user_profile(
            config,
            history_messages,
            messages,
            base_chat_model,
            user_define
        )
        user_profile_data = []
        for profile_type, profile_list in user_profile_memory.items():
            if not isinstance(profile_list, list):
                logger.warning(f"User profile extractor output format error: {profile_list} is not a list")
                continue
            for profile in profile_list:
                input_memories, input_memories_map, input_memory_ids_map = _get_conflict_input(
                    user_id,
                    app_id,
                    profile,
                    self._search_manager
                )
                tmp_conflict_info = ConflictResolution.check_conflict(old_messages=input_memories,
                                                                      new_message=profile,
                                                                      base_chat_model=base_chat_model,
                                                                      config=config)
                conflict_info = _process_conflict_info(tmp_conflict_info, input_memory_ids_map)
                user_profile_data.append(UserProfileUnit(
                    user_id=user_id,
                    app_id=app_id,
                    profile_type=profile_type,
                    profile_mem=profile,
                    conflict_info=conflict_info,
                    mem_type=MemoryType.USER_PROFILE,
                    message_mem_id=message_mem_id
                ))
        return user_profile_data

    def _categories_to_memory_unit(self,
                                   categories: list[str],
                                   history_messages: list[BaseMessage],
                                   messages: list[BaseMessage],
                                   user_id: str,
                                   app_id: str,
                                   config: Config,
                                   base_chat_model: BaseModelClient,
                                   message_mem_id: str,
                                   user_define: dict[str, str] = None
                                   ) -> list[BaseMemoryUnit]:
        memory_units = []
        for category in categories:
            if category not in category_to_class.keys():
                logger.warning(f"Unsupported memory category: {category}, skipped.")
                continue
            mem_class = category_to_class[category]
            if mem_class == MemoryType.USER_PROFILE:
                user_profile_units = self.gen_user_profile(
                    user_id=user_id,
                    app_id=app_id,
                    history_messages=history_messages,
                    messages=messages,
                    config=config,
                    base_chat_model=base_chat_model,
                    message_mem_id=message_mem_id,
                    user_define=user_define
                )
                memory_units += user_profile_units
        return memory_units
