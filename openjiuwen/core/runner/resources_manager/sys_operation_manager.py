#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from typing import Optional, Dict

from openjiuwen.core.sys_operation.sys_operation import SysOperation, SysOperationCard


class SysOperationMgr:
    """Manager for SysOperation instances"""

    def __init__(self):
        self._operations: Dict[str, SysOperation] = {}

    def add_sys_operation(self, card: SysOperationCard) -> SysOperation:
        """
        Add a sys_operation instance.

        Args:
            card: SysOperationCard containing configuration

        Returns:
            SysOperation: Created operation instance

        Raises:
            ValueError: If operation with same ID already exists
        """
        operation_id = card.id
        if operation_id in self._operations:
            raise ValueError(f"SysOperation with id '{operation_id}' already exists")

        # Create SysOperation instance from card
        operation = SysOperation(card)
        self._operations[operation_id] = operation

        return operation

    def remove_sys_operation(self, operation_id: str) -> None:
        """
        Remove a sys_operation instance by ID.

        Args:
            operation_id: Unique identifier for the operation

        Raises:
            KeyError: If operation with specified ID does not exist
        """
        if operation_id in self._operations:
            del self._operations[operation_id]
        else:
            raise KeyError(f"SysOperation with id '{operation_id}' not found")

    def get_sys_operation(self, operation_id: str) -> Optional[SysOperation]:
        """
        Get a sys_operation instance by ID.

        Args:
            operation_id: Unique identifier for the operation

        Returns:
            SysOperation instance if found, None otherwise
        """
        return self._operations.get(operation_id)

    def list_sys_operations(self) -> Dict[str, SysOperation]:
        """
        List all registered sys_operations.

        Returns:
            Dictionary mapping operation IDs to SysOperation instances
        """
        return self._operations.copy()
