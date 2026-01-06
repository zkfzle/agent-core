# !/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import Tuple

from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.generation.long_term_memory_extractor import LongTermMemoryExtractor
from openjiuwen.core.memory.generation.memory_analyzer import MemoryAnalyzer, VariableResult
from openjiuwen.core.memory.manage.data_id_manager import DataIdManager
from openjiuwen.core.memory.mem_unit.memory_unit import MemoryType, BaseMemoryUnit, VariableUnit, UserProfileUnit, \
    SummaryUnit, SemanticMemoryUnit, EpisodicMemoryUnit
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.llm.messages import BaseMessage

category_to_class = {
    "user_profile": MemoryType.USER_PROFILE,
    "semantic_memory": MemoryType.SEMANTIC_MEMORY,
    "episodic_memory": MemoryType.EPISODIC_MEMORY
}


class Generator:
    def __init__(self,
                 data_id_generator: DataIdManager):
        self.data_id_generator = data_id_generator

    async def gen_all_memory(self, **kwargs) -> list[BaseMemoryUnit]:
        """Generate all memory units based on input"""
        messages = kwargs.get("messages")
        config = kwargs.get("config")
        model = kwargs.get("base_chat_model")
        user_id = kwargs.get("user_id")
        group_id = kwargs.get("group_id")
        history_messages = kwargs.get("history_messages")
        message_mem_id = kwargs.get("message_mem_id")
        timestamp = kwargs.get("timestamp")

        memory_analyze_res = await MemoryAnalyzer.analyze(
            messages=messages,
            history_messages=history_messages,
            base_chat_model=model,
            memory_config=config,
        )
        all_memory_results = []
        variable_units = Generator._process_extracted_data(
            user_id=user_id,
            group_id=group_id,
            variable_results=memory_analyze_res.variables,
        )
        all_memory_results += variable_units

        summary_unit = await self._process_summary_data(user_id=user_id,
                                                        group_id=group_id,
                                                        message_mem_id=message_mem_id,
                                                        summary=memory_analyze_res.summary,
                                                        timestamp=timestamp)
        all_memory_results.append(summary_unit)
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
                message_mem_id=message_mem_id,
                timestamp=timestamp,
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

    @staticmethod
    def _process_extracted_data(
            user_id: str,
            group_id: str,
            variable_results: list[VariableResult],
    ) -> list[VariableUnit]:
        variable_units = []
        for tmp_data in variable_results:
            if not tmp_data.variable_value:
                continue
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
                                         timestamp: str
                                         ) -> list[BaseMemoryUnit]:
        memory_units = []
        memory_dict = await LongTermMemoryExtractor.extract_long_term_memory(
            categories=categories,
            history_messages=history_messages,
            messages=messages,
            base_chat_model=base_chat_model,
            timestamp=timestamp,
        )
        memory_units.extend(await self._get_user_profile_unit(
            user_id=user_id,
            group_id=group_id,
            message_mem_id=message_mem_id,
            memory_dict=memory_dict,
            timestamp=timestamp,
        ))
        memory_units.extend(await self._get_semantic_memory_unit(
            user_id=user_id,
            group_id=group_id,
            message_mem_id=message_mem_id,
            memory_dict=memory_dict,
            timestamp=timestamp,
        ))
        memory_units.extend(await self._get_episodic_memory_unit(
            user_id=user_id,
            group_id=group_id,
            message_mem_id=message_mem_id,
            memory_dict=memory_dict,
            timestamp=timestamp,
        ))
        return memory_units

    async def _process_summary_data(
            self,
            user_id: str,
            group_id: str,
            message_mem_id: str,
            summary: str,
            timestamp: str,
    ) -> SummaryUnit:
        mem_id = str(await self.data_id_generator.generate_next_id(user_id=user_id))
        return SummaryUnit(
            user_id=user_id,
            group_id=group_id,
            mem_type=MemoryType.SUMMARY,
            mem_id=mem_id,
            summary=summary,
            message_mem_id=message_mem_id,
            timestamp=timestamp,
        )

    async def _get_user_profile_unit(
            self,
            user_id: str,
            group_id: str,
            message_mem_id: str,
            memory_dict: dict,
            timestamp: str
    ) -> list[UserProfileUnit]:
        """Generate user profile memory unit based on input"""
        user_profile_data = []
        user_profile_dict = memory_dict.get("user_profile", {})
        for profile_type, profile_list in user_profile_dict.items():
            if not isinstance(profile_list, list):
                logger.warning(f"User profile extractor output format error: {profile_list} is not a list")
                continue
            for profile in profile_list:
                mem_id = str(await self.data_id_generator.generate_next_id(user_id=user_id))
                user_profile_data.append(UserProfileUnit(
                    user_id=user_id,
                    group_id=group_id,
                    profile_type=profile_type,
                    profile_mem=profile,
                    mem_type=MemoryType.USER_PROFILE,
                    message_mem_id=message_mem_id,
                    timestamp=timestamp,
                    mem_id=mem_id,
                ))
        return user_profile_data

    async def _get_semantic_memory_unit(
            self,
            user_id: str,
            group_id: str,
            message_mem_id: str,
            memory_dict: dict,
            timestamp: str
    ) -> list[SemanticMemoryUnit]:
        """"""
        semantic_memory_units = []
        semantic_memory_list = memory_dict.get("semantic_memory", [])
        if isinstance(semantic_memory_list, list) and len(semantic_memory_list) > 0:
            for memory in semantic_memory_list:
                if not isinstance(memory, str):
                    logger.warning(f"semantic memory format error: {memory} is not a list")
                    continue
                mem_id = str(await self.data_id_generator.generate_next_id(user_id=user_id))
                semantic_memory_units.append(SemanticMemoryUnit(
                    user_id=user_id,
                    group_id=group_id,
                    mem_type=MemoryType.SEMANTIC_MEMORY,
                    semantic_mem=memory,
                    message_mem_id=message_mem_id,
                    timestamp=timestamp,
                    mem_id=mem_id,
                ))
        return semantic_memory_units

    async def _get_episodic_memory_unit(
            self,
            user_id: str,
            group_id: str,
            message_mem_id: str,
            memory_dict: dict,
            timestamp: str
    ) -> list[EpisodicMemoryUnit]:
        """Generate episodic memory unit based on input"""
        episodic_memory_units = []
        episodic_memory_list = memory_dict.get("episodic_memory", [])
        if isinstance(episodic_memory_list, list) and len(episodic_memory_list) > 0:
            for episodic_memory in episodic_memory_list:
                if not isinstance(episodic_memory, str):
                    logger.warning(f"episodic memory format error: {episodic_memory} is not a list")
                    continue

                mem_id = str(await self.data_id_generator.generate_next_id(user_id=user_id))
                episodic_memory_units.append(EpisodicMemoryUnit(
                    mem_type=MemoryType.EPISODIC_MEMORY,
                    user_id=user_id,
                    group_id=group_id,
                    content=episodic_memory,
                    message_mem_id=message_mem_id,
                    timestamp=timestamp,
                    mem_id=mem_id,
                ))
        return episodic_memory_units
