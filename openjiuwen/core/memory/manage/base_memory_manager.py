#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from abc import abstractmethod, ABC
from typing import Any

from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.common.crypto import encrypt, decrypt, IV_LENGTH

from openjiuwen.core.memory.mem_unit.memory_unit import BaseMemoryUnit


class BaseMemoryManager(ABC):
    """
    Simplified abstract base class for memory manager implementations.
    Managing a specific type of memory data.
    """

    IV_HEX_LENGTH = IV_LENGTH * 2 # hex_length = bytes_length * 2

    @abstractmethod
    async def add(self, memory: BaseMemoryUnit):
        """add memory."""
        pass

    @abstractmethod
    async def update(self, user_id: str, group_id: str, mem_id: str, new_memory: str, **kwargs):
        """update memory by its id."""
        pass

    @abstractmethod
    async def delete(self, user_id: str, group_id: str, mem_id: str, **kwargs):
        """delete memory by its id."""
        pass

    @abstractmethod
    async def delete_by_user_id(self, user_id: str, group_id: str):
        """delete memory by user id and app id."""
        pass

    @abstractmethod
    async def get(self, user_id: str, group_id: str, mem_id: str) -> dict[str, Any] | None:
        """get memory by its id."""
        pass

    @abstractmethod
    async def search(self, user_id: str, group_id: str, query: str, top_k: int, **kwargs):
        """query memory, return top k results"""
        pass

    @staticmethod
    def encrypt_memory_if_needed(key: str, plaintext: str) -> str:
        if not key or not plaintext:
            return plaintext

        try:
            encrypt_memory, iv = encrypt(key=key, plaintext=plaintext)
            return f"{iv}{encrypt_memory}"
        except ValueError as e:
            logger.warning(f"Encrypt exception occurred:{str(e)}")
            return ""
        except Exception as e:
            logger.warning(f"Encrypt error occurred:{str(e)}")
            return ""

    @staticmethod
    def decrypt_memory_if_needed(key: str, ciphertext: str) -> str:
        if not key or not ciphertext:
            return ciphertext

        if len(ciphertext) < BaseMemoryManager.IV_HEX_LENGTH:
            logger.warning(f"Decryption error occurred: invalid ciphertext len{len(ciphertext)}")
            return ""

        iv = ciphertext[0:BaseMemoryManager.IV_HEX_LENGTH]
        encrypt_memory = ciphertext[BaseMemoryManager.IV_HEX_LENGTH:]
        try:
            return decrypt(key=key, ciphertext=encrypt_memory, iv=iv)
        except ValueError as e:
            logger.warning(f"Decrypt exception occurred:{str(e)}")
            return ""
        except Exception as e:
            logger.warning(f"Decrypt error occurred:{str(e)}")
            return ""
