#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from sqlalchemy import Column, String, Engine
from sqlalchemy.orm import declarative_mixin, declarative_base

Base = declarative_base()


@declarative_mixin
class MessageMixin:
    """Definition of public Field"""
    message_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    app_id = Column(String, nullable=False)
    content = Column(String, nullable=False)
    session_id = Column(String, nullable=True)
    role = Column(String, nullable=True)
    timestamp = Column(String, nullable=True)


class UserMessage(MessageMixin, Base):
    """Unique field and table name definition for user messages table"""
    __tablename__ = "user_message"


def create_tables(engine: Engine):
    Base.metadata.create_all(engine,tables=[UserMessage.__table__])
