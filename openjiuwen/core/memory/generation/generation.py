# !/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import Tuple
from openjiuwen.core.memory.generation.long_term_memory_extractor import LongTermMemoryExtractor
from openjiuwen.core.memory.mem_unit.memory_unit import (MemoryType, BaseMemoryUnit, VariableUnit, UserProfileUnit,
                                                         SemanticMemoryUnit)
from openjiuwen.core.memory.generation.memory_analyzer import MemoryAnalyzer, VariableResult
from openjiuwen.core.common.logging import logger
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.llm.messages import BaseMessage

category_to_class = {
    "user_profile": MemoryType.USER_PROFILE,
    "semantic_memory": MemoryType.SEMANTIC_MEMORY
}


def get_user_profile_unit(
        user_id: str,
        group_id: str,
        message_mem_id: str,
        memory_dict: dict
) -> list[UserProfileUnit]:
    """Generate user profile memory unit based on input"""
    user_profile_data = []
    user_profile_dict = memory_dict.get("user_profile", {})
    for profile_type, profile_list in user_profile_dict.items():
        if not isinstance(profile_list, list):
            logger.warning(f"User profile extractor output format error: {profile_list} is not a list")
            continue
        for profile in profile_list:
            user_profile_data.append(UserProfileUnit(
                user_id=user_id,
                group_id=group_id,
                profile_type=profile_type,
                profile_mem=profile,
                mem_type=MemoryType.USER_PROFILE,
                message_mem_id=message_mem_id,
            ))
    return user_profile_data

def get_semantic_memory_unit(
        user_id: str,
        group_id: str,
        message_mem_id: str,
        memory_dict: dict
) -> list[SemanticMemoryUnit]:
    """"""
    semantic_memory_units = []
    semantic_memory_list = memory_dict.get("semantic_memory", [])
    if isinstance(semantic_memory_list, list) and len(semantic_memory_list) > 0:
        for memory in semantic_memory_list:
            if not isinstance(memory, str):
                logger.warning(f"semantic memory format error: {memory} is not a list")
                continue
            semantic_memory_units.append(SemanticMemoryUnit(
                user_id=user_id,
                group_id=group_id,
                mem_type=MemoryType.SEMANTIC_MEMORY,
                semantic_mem=memory,
                message_mem_id=message_mem_id,
            ))
    return semantic_memory_units

class Generator:
    async def gen_all_memory(self, **kwargs) -> list[BaseMemoryUnit]:
        """Generate all memory units based on input"""
        messages = kwargs.get("messages")
        config = kwargs.get("config")
        model = kwargs.get("base_chat_model")
        user_id = kwargs.get("user_id")
        group_id = kwargs.get("group_id")
        history_messages = kwargs.get("history_messages")
        message_mem_id = kwargs.get("message_mem_id")
        if not all([messages, config, user_id, group_id, model]):
            logger.error("messages, config, user_id, group_id, model are required parameters")

        memory_analyze_res = await MemoryAnalyzer.analyze(
            messages=messages,
            history_messages=history_messages,
            base_chat_model=model,
            memory_config=config,
        )
        all_memory_results = []
        variable_units = self._process_extracted_data(
            user_id=user_id,
            group_id=group_id,
            variable_results=memory_analyze_res.variables,
        )
        all_memory_results += variable_units
        if not config.enable_long_term_mem:
            logger.info("Not enable long term memory")
            return all_memory_results
        try:
            merged_units = await self._categories_to_memory_unit(
                categories=memory_analyze_res.categories,
                history_messages=history_messages,
                messages=messages,
                user_id=user_id,
                group_id=group_id,
                base_chat_model=model,
                message_mem_id=message_mem_id
            )
        except AttributeError as e:
            logger.debug(f"Get conflict info has attribute exception: {str(e)}")
            return all_memory_results
        except ValueError as e:
            logger.warning(f"Get conflict info has value exception: {str(e)}")
            return all_memory_results
        except BaseException as e:
            logger.warning(f"Get conflict info has exception: {str(e)}")
            return all_memory_results
        all_memory_results += merged_units
        return all_memory_results

    def _process_extracted_data(
            self,
            user_id: str,
            group_id: str,
            variable_results: list[VariableResult],
    ) -> list[VariableUnit]:
        variable_units = []
        for tmp_data in variable_results:
            variable_units.append(VariableUnit(
                user_id=user_id,
                group_id=group_id,
                mem_type=MemoryType.VARIABLE,
                variable_name=tmp_data.variable_key,
                variable_mem=tmp_data.variable_value
            ))
        return variable_units

    async def _categories_to_memory_unit(self,
                                         categories: list[str],
                                         history_messages: list[BaseMessage],
                                         messages: list[BaseMessage],
                                         user_id: str,
                                         group_id: str,
                                         base_chat_model: Tuple[str, BaseModelClient],
                                         message_mem_id: str,
                                         user_define: dict[str, str] = None
                                         ) -> list[BaseMemoryUnit]:
        memory_units = []
        memory_dict = await LongTermMemoryExtractor.extract_long_term_memory(
            categories=categories,
            history_messages=history_messages,
            messages=messages,
            base_chat_model=base_chat_model,
        )
        memory_units.extend(get_user_profile_unit(
            user_id=user_id,
            group_id=group_id,
            message_mem_id=message_mem_id,
            memory_dict=memory_dict
        ))
        memory_units.extend(get_semantic_memory_unit(
            user_id=user_id,
            group_id=group_id,
            message_mem_id=message_mem_id,
            memory_dict=memory_dict
        ))
        return memory_units
