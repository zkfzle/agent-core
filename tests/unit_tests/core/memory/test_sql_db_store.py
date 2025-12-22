#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import asyncio
import logging
import unittest
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from openjiuwen.core.memory.store.impl.default_db_store import DefaultDbStore
from openjiuwen.core.memory.store.sql_db_store import SqlDbStore

logger = logging.getLogger(__name__)


class ContextStoreColumnType(StrEnum):
    TEXT = 'TEXT'
    INTEGER = 'INTEGER'
    REAL = 'REAL'
    BLOB = 'BLOB'
    NUMERIC = 'NUMERIC'


CONTEXT_CONFIG = {
    'table': 'user_messages',
    'columns': {
        'user_id': ContextStoreColumnType.TEXT,
        'group_id': ContextStoreColumnType.TEXT,
        'session_id': ContextStoreColumnType.TEXT,
        'message_id': ContextStoreColumnType.TEXT,
        'role': ContextStoreColumnType.TEXT,
        'content': ContextStoreColumnType.TEXT,
        'timestamp': ContextStoreColumnType.TEXT
    }
}

data_list = [
    {
        "user_id": "u1",
        "group_id": "group1",
        "session_id": "s1",
        "message_id": "m1",
        "role": "user",
        "content": "Hello",
        "timestamp": "2025-11-19 09:00:00"
    },
    {
        "user_id": "u1",
        "group_id": "group1",
        "session_id": "s1",
        "message_id": "m2",
        "role": "user",
        "content": "World",
        "timestamp": "2025-11-19 10:00:00"
    },
    {
        "user_id": "u2",
        "group_id": "group2",
        "session_id": "s2",
        "message_id": "m3",
        "role": "assistant",
        "content": "Hi there",
        "timestamp": "2025-11-19 11:00:00"
    }
]


async def create(conn: AsyncEngine, table: str, columns: dict[str, ContextStoreColumnType]):
    try:
        async with conn.begin() as conn:
            await conn.execute(text(
                f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY)"
            ))
            cursor = await conn.execute(text(
                f"PRAGMA table_info('{table}')"
            ))
            existing_items = {row[1] for row in cursor.fetchall()}
            for column_name, column_type in columns.items():
                if column_name in existing_items:
                    continue
                alter_sql = f"ALTER TABLE {table} ADD COLUMN '{column_name}' {column_type}"
                await conn.execute(text(alter_sql))
            await conn.commit()
    except Exception as e:
        logger.error("Failed to create table", exc_info=e)


class TestAsyncSqlDbStore(unittest.TestCase):
    def setUp(self):
        utc_now = datetime.now(timezone.utc)
        local_time = utc_now.replace(tzinfo=None)
        time_str = local_time.strftime("%Y%m%d%H%M%S")
        uuid_str = uuid.uuid4().hex[:6]
        self.path = Path(f"./test_sql_db_{time_str}_{uuid_str}.db").resolve()
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{self.path}")
        db_store = DefaultDbStore(self.engine)
        asyncio.run(create(db_store.get_async_engine(), CONTEXT_CONFIG['table'], CONTEXT_CONFIG['columns']))
        self.store = SqlDbStore(db_store)

    def tearDown(self):
        asyncio.run(self.engine.dispose())
        if self.path.exists():
            self.path.unlink()

    async def async_get_table_columns(self):
        """Verify that _get_table correctly retrieves the table schema."""
        table_name = CONTEXT_CONFIG["table"]
        table = await self.store.get_table(table_name)

        expected_cols = list(CONTEXT_CONFIG["columns"].keys())

        # Verify that all columns exist.
        for col in expected_cols:
            self.assertIn(col, table.c)

        # Verify that each column type is correctly loaded.
        for col in table.c:
            self.assertTrue(hasattr(col.type, "python_type"))

    async def async_add(self):
        inner_data_list = [
            {
                "user_id": "u1",
                "group_id": "group1",
                "session_id": "s1",
                "message_id": "m1",
                "role": "user",
                "content": "Hello",
                "timestamp": "2025-11-19 09:00:00"
            },
            {
                "user_id": "u1",
                "group_id": "group1",
                "session_id": "s1",
                "message_id": "m2",
                "role": "user",
                "content": "World",
                "timestamp": "2025-11-19 10:00:00"
            },
            {
                "user_id": "u2",
                "group_id": "group2",
                "session_id": "s2",
                "message_id": "m3",
                "role": "assistant",
                "content": "Hi there",
                "timestamp": "2025-11-19 11:00:00"
            }
        ]

        for data in inner_data_list:
            self.assertTrue(await self.store.write(CONTEXT_CONFIG["table"], data))

    async def async_get(self):
        filters = {}
        filters['message_id'] = ["m1"]
        row = await self.store.condition_get(
            CONTEXT_CONFIG["table"],
            conditions=filters
        )
        self.assertIsNotNone(row)
        self.assertEqual(row[0]["content"], "Hello")
        self.assertEqual(row[0]["user_id"], "u1")

    async def async_get_with_sort(self):
        """Verify that get_with_sort correctly performs query sorting."""

        table = CONTEXT_CONFIG["table"]

        rows = await self.store.get_with_sort(
            table=table,
            filters={"user_id": "u1"},
            sort_by="timestamp",
            order="ASC",
            limit=10
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["message_id"], "m1")  # 最旧
        self.assertEqual(rows[1]["message_id"], "m2")  # 最新

        # user_id = u2
        rows_u2 = await self.store.get_with_sort(
            table=table,
            filters={"user_id": "u2"},
            sort_by="timestamp",
            order="ASC",
            limit=10
        )

        self.assertTrue(len(rows_u2) >= 1)
        self.assertEqual(rows_u2[0]["message_id"], "m3")

    async def async_exist(self):
        all_data = await self.store.get_with_sort(
            table=CONTEXT_CONFIG["table"], filters={})
        logger.info(f"all_data: {all_data}")
        self.assertGreater(len(all_data), 0)
        self.assertTrue(await self.store.exist(CONTEXT_CONFIG["table"], {"message_id": "m1"}))
        self.assertFalse(await self.store.exist(CONTEXT_CONFIG["table"], {"message_id": "not_exist"}))
        self.assertTrue(await self.store.exist(CONTEXT_CONFIG["table"], {"user_id": "u1", "content": "Hello"}))
        self.assertFalse(await self.store.exist(CONTEXT_CONFIG["table"], {"user_id": "u1", "content": "foo"}))
        self.assertFalse(await self.store.exist(CONTEXT_CONFIG["table"], {"user_id": "uX", "content": "bar"}))

    async def async_update(self):
        ok = await self.store.update(CONTEXT_CONFIG["table"], {"message_id": "m1"}, {"content": "hi"})
        self.assertTrue(ok)
        row = await self.store.condition_get(CONTEXT_CONFIG["table"], {"message_id": ["m1"]})
        self.assertEqual(row[0]["content"], "hi")
        ok = await self.store.update(CONTEXT_CONFIG["table"], {"message_id": ["m2", "m3"]}, {"content": "batch"})
        self.assertTrue(ok)
        row2 = await self.store.condition_get(CONTEXT_CONFIG["table"], {"message_id": ["m2"]})
        row3 = await self.store.condition_get(CONTEXT_CONFIG["table"], {"message_id": ["m3"]})
        self.assertEqual(row2[0]["content"], "batch")
        self.assertEqual(row3[0]["content"], "batch")

    async def async_delete(self):
        ok = await self.store.delete(CONTEXT_CONFIG["table"], {"message_id": "m1"})
        self.assertTrue(ok)
        row = await self.store.condition_get(CONTEXT_CONFIG["table"], {"id": ["m1"]})
        self.assertEqual(row, [])

    @unittest.skip("skip test")
    def test_basic(self):
        asyncio.run(self.async_get_table_columns())
        asyncio.run(self.async_add())
        asyncio.run(self.async_get())
        asyncio.run(self.async_get_with_sort())
        asyncio.run(self.async_exist())
        asyncio.run(self.async_update())
        asyncio.run(self.async_delete())


if __name__ == "__main__":
    unittest.main()
