#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

from openjiuwen.core.memory.store.sql_db_store import SqlDbStore
from openjiuwen.core.memory.manage.data_id_manager import DataIdManager
from openjiuwen.core.utils.llm.messages import BaseMessage


## DB-Based Message Management
class MessageManager:
    def __init__(self, sql_db_store: SqlDbStore, data_id_manager: DataIdManager):
        self.sql_db = sql_db_store
        self.message_table = "user_message"
        self.data_id = data_id_manager

    def add(self, user_id: str = None, app_id: str = None, content: str = None,
            role: str = None, session_id: str = None, timestamp: datetime = None) -> str:
        message_id = str(self.data_id.generate_next_id())
        if user_id is None:
            raise ValueError('Must provide user_id')
        if app_id is None:
            raise ValueError('Must provide app_id')
        if content is None:
            raise ValueError('Must provide content')
        time = datetime.now(timezone.utc) if not timestamp else timestamp
        data = {
            'message_id': message_id,
            'user_id': user_id or '',
            'session_id': session_id or '',
            'app_id': app_id or '',
            'role': role or '',
            'content': content,
            'timestamp': time
        }
        self.sql_db.write(self.message_table, data)
        return message_id

    def get(self, user_id: str = None, app_id: str = None, session_id: str = None,
            message_len: int = 10) -> list[Tuple[BaseMessage, datetime]]:
        filters: Dict[str, Any] = {}
        if user_id is not None:
            filters['user_id'] = user_id
        if app_id is not None:
            filters['app_id'] = app_id
        if session_id is not None:
            filters['session_id'] = session_id
        if message_len <= 0:
            raise ValueError('message_len Must bigger than zero')
        messages = self.sql_db.get_with_sort(table=self.message_table, filters=filters, limit=message_len)
        return [(BaseMessage(**message), message['timestamp']) for message in messages]

    def get_by_id(self, msg_id: str) -> list[Tuple[BaseMessage, datetime]]:
        filters: Dict[str, Any] = {}
        filters['message_id'] = [msg_id]
        messages = self.sql_db.condition_get(table=self.message_table, conditions=filters)
        return [(BaseMessage(**message), message['timestamp']) for message in messages]
