#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import asyncio
from pathlib import Path
import unittest

from sqlalchemy import text, inspect, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncEngine
from openjiuwen.core.memory.store.message import create_tables, UserMessage


@unittest.skip("skip test")
class TestCreateTable(unittest.TestCase):
    def test_table_creation(self):
        path = Path("./memory_engine.db")
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{path.resolve()}"
        )
        asyncio.run(create_tables(engine))
        asyncio.run(self._async_check_table(engine))

        asyncio.run(TestCreateTable._clear_tables(engine))

        asyncio.run(self._insert_data(engine))

    async def _insert_data(self, engine: AsyncEngine):
        async_session = async_sessionmaker(bind=engine, expire_on_commit=False)
        msg = UserMessage(
            user_id="u123",
            app_id="app456",
            session_id="s789",
            message_id="m001",
            role="user",
            content="hello",
            timestamp="2025-11-18 19:00:00",
        )
        async with async_session() as session:
            async with session.begin():
                session.add(msg)
                await session.commit()

        async with async_session() as session:
            async with session.begin():
                result = await session.execute(select(UserMessage))
                messages = result.scalars().all()
                for m in messages:
                    print(m.message_id, m.user_id, m.content, m.timestamp)
                    self.assertEqual(m.message_id, msg.message_id)
                    self.assertEqual(m.user_id, msg.user_id)
                    self.assertEqual(m.content, msg.content)
                    self.assertEqual(m.timestamp, msg.timestamp)
                    self.assertEqual(m.role, msg.role)
                    self.assertEqual(m.session_id, msg.session_id)
                    self.assertEqual(m.app_id, msg.app_id)

    @staticmethod
    async def _clear_tables(engine: AsyncEngine):
        async with engine.connect() as conn:
            await conn.execute(text(f"DELETE FROM {UserMessage.__tablename__}"))
            await conn.commit()

    async def _async_check_table(self, async_engine):
        async with async_engine.connect() as conn:
            def sync_reflect(sync_conn):
                ins = inspect(sync_conn)
                t = ins.get_table_names()
                return t

            table = await conn.run_sync(sync_reflect)
            self.assertIn(UserMessage.__tablename__, table)