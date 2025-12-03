#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from sqlalchemy import Column, String
from sqlalchemy.orm import declarative_mixin, declarative_base

from openjiuwen.core.memory.store.base_db_store import BaseDbStore

Base = declarative_base()


@declarative_mixin
class MessageMixin:
    """Definition of public Field"""
    message_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    group_id = Column(String, nullable=False)
    content = Column(String, nullable=False)
    session_id = Column(String, nullable=True)
    role = Column(String, nullable=True)
    timestamp = Column(String, nullable=True)


class UserMessage(MessageMixin, Base):
    """Unique field and table name definition for user messages table"""
    __tablename__ = "user_message"


async def create_tables(db_store: BaseDbStore):
    async with db_store.get_async_engine().begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(
                sync_conn,
                tables=[UserMessage.__table__]
            )
        )