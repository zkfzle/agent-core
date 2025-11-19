from openjiuwen.core.memory.store.sql_db_store import SqlDbStore
from sqlalchemy import engine, QueuePool, create_engine, text, Table
import unittest
from pathlib import Path
from enum import StrEnum
import logging
from typing import Dict, Any, List
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
        'app_id': ContextStoreColumnType.TEXT,
        'session_id': ContextStoreColumnType.TEXT,
        'message_id': ContextStoreColumnType.TEXT,
        'role': ContextStoreColumnType.TEXT,
        'content': ContextStoreColumnType.TEXT,
        'timestamp': ContextStoreColumnType.TEXT
    }
}

def create(conn: engine.Engine, table: str, columns: dict[str, ContextStoreColumnType]):
    try:
        with conn.connect() as conn:
            conn.execute(text(
                f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY)"
            ))
            cursor = conn.execute(text(
                f"PRAGMA table_info('{table}')"
            ))
            existing_items = {row[1] for row in cursor.fetchall()}
            for column_name, column_type in columns.items():
                if column_name in existing_items:
                    continue
                alter_sql = f'ALTER TABLE {table} ADD COLUMN "{column_name}" {column_type}'
                conn.execute(text(alter_sql))
            conn.commit()
    except Exception as e:
        logger.error("Failed to create table", exc_info=e)

class TestSqlDbStore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        整个测试类只初始化一次数据库，确保所有测试复用相同的数据表。
        """
        cls.db_path = Path("./sql_db.db")
        if cls.db_path.exists():
            cls.db_path.unlink()  # 删除旧文件

        cls.engine = create_engine(
            f"sqlite:///{cls.db_path.resolve()}",
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

        # 创建表
        create(cls.engine, CONTEXT_CONFIG["table"], CONTEXT_CONFIG["columns"])
        cls.store = SqlDbStore(cls.engine)

        data_list = [
            {
                "user_id": "u1",
                "app_id": "app1",
                "session_id": "s1",
                "message_id": "m1",
                "role": "user",
                "content": "Hello",
                "timestamp": "2025-11-19 09:00:00"
            },
            {
                "user_id": "u1",
                "app_id": "app1",
                "session_id": "s1",
                "message_id": "m2",
                "role": "user",
                "content": "World",
                "timestamp": "2025-11-19 10:00:00"
            },
            {
                "user_id": "u2",
                "app_id": "app2",
                "session_id": "s2",
                "message_id": "m3",
                "role": "assistant",
                "content": "Hi there",
                "timestamp": "2025-11-19 11:00:00"
            }
        ]

        for data in data_list:
            cls.store.write(CONTEXT_CONFIG["table"], data)

    def test_get_table_columns(self):
        """测试 _get_table 是否能正确读取表结构"""
        table_name = CONTEXT_CONFIG["table"]
        table: Table = self.store._get_table(table_name)

        expected_cols = list(CONTEXT_CONFIG["columns"].keys())

        # 检查所有列是否存在
        for col in expected_cols:
            self.assertIn(col, table.c)

        # 检查每个列类型能正确加载
        for col in table.c:
            self.assertTrue(hasattr(col.type, "python_type"))

    def test_get_with_sort(self):
        """测试 get_with_sort 查询排序是否正确"""

        table = CONTEXT_CONFIG["table"]

        rows = self.store.get_with_sort(
            table=table,
            filters={"user_id": "u1"},
            sort_by="timestamp",
            order="DESC",
            limit=10
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["message_id"], "m2")  # 最新
        self.assertEqual(rows[1]["message_id"], "m1")  # 最旧

        # user_id = u2
        rows_u2 = self.store.get_with_sort(
            table=table,
            filters={"user_id": "u2"},
            sort_by="timestamp",
            order="DESC",
            limit=10
        )

        self.assertEqual(len(rows_u2), 1)
        self.assertEqual(rows_u2[0]["message_id"], "m3")

    def test_get(self):
        filters = {}
        filters['message_id'] = ["m1"]
        row = self.store.condition_get(
            CONTEXT_CONFIG["table"],
            conditions=filters
        )
        self.assertIsNotNone(row)
        self.assertEqual(row[0]["content"], "Hello")
        self.assertEqual(row[0]["user_id"], "u1")

    def test_exist(self):
        self.assertTrue(self.store.exist(CONTEXT_CONFIG["table"], {"message_id": "m1"}))
        self.assertFalse(self.store.exist(CONTEXT_CONFIG["table"], {"message_id": "not_exist"}))
        self.assertTrue(self.store.exist(CONTEXT_CONFIG["table"], {"user_id": "u1", "content": "Hello"}))
        self.assertFalse(self.store.exist(CONTEXT_CONFIG["table"], {"user_id": "u1", "content": "foo"}))
        self.assertFalse(self.store.exist(CONTEXT_CONFIG["table"], {"user_id": "uX", "content": "bar"}))

    def test_update(self):
        ok = self.store.update(CONTEXT_CONFIG["table"], {"message_id": "m1"}, {"content": "hi"})
        self.assertTrue(ok)
        row = self.store.condition_get(CONTEXT_CONFIG["table"], {"message_id": ["m1"]})
        self.assertEqual(row[0]["content"], "hi")
        ok = self.store.update(CONTEXT_CONFIG["table"], {"message_id": ["m2", "m3"]}, {"content": "batch"})
        self.assertTrue(ok)
        row2 = self.store.condition_get(CONTEXT_CONFIG["table"], {"message_id": ["m2"]})
        row3 = self.store.condition_get(CONTEXT_CONFIG["table"], {"message_id": ["m3"]})
        self.assertEqual(row2[0]["content"], "batch")
        self.assertEqual(row3[0]["content"], "batch")

    def test_delete(self):
        ok = self.store.delete(CONTEXT_CONFIG["table"], {"message_id": "m1"})
        self.assertTrue(ok)
        row = self.store.condition_get("messages", {"id": ["m1"]})
        self.assertIsNone(row)

    def test_close(self):
        self.store.close()

    def test_delete_table(self):
        self.store.delete_table(CONTEXT_CONFIG["table"])

if __name__ == "__main__":
    unittest.main()
