# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from typing import Optional, Dict
from openjiuwen.core.runner.resources_manager.abstract_manager import AbstractManager
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.sys_operation.sys_operation import SysOperation, SysOperationCard


class SysOperationMgr(AbstractManager[SysOperation]):
    """Manager for SysOperation instances"""

    def __init__(self):
        super().__init__()

    def add_sys_operation(self, card: SysOperationCard) -> SysOperation:
        """
        Add a sys_operation instance.

        Args:
            card: SysOperationCard containing configuration

        Returns:
            SysOperation: Created operation instance
        """
        operation_id = card.id
        self._validate_id(operation_id, StatusCode.SESSION_SYS_OP_ADD_FAILED, "sys_operation")

        # Create validate function for add_resource
        def validate_sys_op(op_card):
            if operation_id in self._resources:
                from openjiuwen.core.common.exception.exception import JiuWenBaseException
                raise JiuWenBaseException(
                    StatusCode.SESSION_SYS_OP_ADD_FAILED.code,
                    StatusCode.SESSION_SYS_OP_ADD_FAILED.errmsg.format(
                        reason=f"SysOperation with id '{operation_id}' already exists")
                )
            return SysOperation(op_card)

        self._add_resource(operation_id, card, StatusCode.SESSION_SYS_OP_ADD_FAILED, validate_sys_op)
        return self._resources.get(operation_id)

    def remove_sys_operation(self, operation_id: str) -> Optional[SysOperation]:
        """
        Remove a sys_operation instance by ID.

        Args:
            operation_id: Unique identifier for the operation

        Returns:
            The removed SysOperation instance, or None if it didn't exist (and skip_if_not_exists is True)
        """
        self._validate_id(operation_id, StatusCode.SESSION_SYS_OP_REMOVE_FAILED, "sys_operation")
        return self._remove_resource(operation_id, StatusCode.SESSION_SYS_OP_REMOVE_FAILED)

    def get_sys_operation(self, operation_id: str) -> Optional[SysOperation]:
        """
        Get a sys_operation instance by ID.

        Args:
            operation_id: Unique identifier for the operation

        Returns:
            SysOperation instance if found, None otherwise
        """
        self._validate_id(operation_id, StatusCode.SESSION_SYS_OP_GET_FAILED, "sys_operation")
        return self._get_resource(operation_id, StatusCode.SESSION_SYS_OP_GET_FAILED)
