from pathlib import Path
import unittest

from sqlalchemy import create_engine, QueuePool, text, inspect
from sqlalchemy.orm import sessionmaker

from openjiuwen.core.memory.store.message import create_tables, UserMessage


class TestCreateTable(unittest.TestCase):
    def test_table_creation(self):
        path = Path("./memory_engine.db")
        engine = create_engine(
            f"sqlite:///{path.resolve()}",
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        create_tables(engine)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        self.assertIn(UserMessage.__tablename__, tables)

        with engine.connect() as conn:
            conn.execute(text(f"DELETE FROM {UserMessage.__tablename__}"))
            conn.commit()

        Session = sessionmaker(bind=engine)
        session = Session()
        msg = UserMessage(
            user_id="u123",
            app_id="app456",
            session_id="s789",
            message_id="m001",
            role="user",
            content="hello",
            timestamp="2025-11-18 19:00:00",
        )
        session.add(msg)
        session.commit()
        for m in session.query(UserMessage).all():
            print(m.message_id, m.user_id, m.content, m.timestamp)
            self.assertEquals(msg, m)
