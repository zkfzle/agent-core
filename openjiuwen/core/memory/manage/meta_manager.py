from openjiuwen.core.memory.store.sql_db_store import SqlDbStore
from typing import Any

class MetaManager:
    def __init__(self,
                 sql_db_store: SqlDbStore):
        self.sql_db = sql_db_store
        self.meta_table = "meta_data"

    async def add(self, user_id: str, scope_id: str, **kwargs):
        data = {
            'user_id': user_id or '',
            'scope_id': scope_id or '',
        }
        exists = await self.sql_db.exist(
            table=self.meta_table,
            conditions={
                "user_id": data["user_id"],
                "scope_id": data["scope_id"],
            },
        )
        if exists:
            return
        await self.sql_db.write(self.meta_table, data)

    async def delete_by_scope_id(self, scope_id: str) -> bool:
        return await self.sql_db.delete(
            table=self.meta_table,
            conditions={"scope_id": scope_id}
        )

    async def get_by_scope_id(self, scope_id: str) -> dict[str, Any] | None:
        results = await self.sql_db.condition_get(
            table=self.meta_table,
            conditions={"scope_id": [scope_id]},
            columns=None
        )
        return results if results else None

if __name__ == "__main__":
    import os
    import asyncio
    from sqlalchemy.ext.asyncio import create_async_engine
    from openjiuwen.core.memory.store.impl.default_db_store import DefaultDbStore
    user_id = "user0108_2"
    scope_id = "app0108_2"
    db_user = os.getenv("DB_USER", "root")
    db_passport = os.getenv("DB_PASSWORD", "root")
    db_host = os.getenv("DB_HOST", "124.71.229.79")
    db_port = os.getenv("DB_PORT", "33306")
    agent_db_name = os.getenv("AGENT_DB_NAME", "jiuwen_agent")

    db_store = DefaultDbStore(create_async_engine(
        f"mysql+aiomysql://{db_user}:{db_passport}@{db_host}:{db_port}/{agent_db_name}?charset=utf8mb4",
        pool_size=20,
        max_overflow=20
    ))
    sql_db_store = SqlDbStore(db_store)
    meta_manager = MetaManager(sql_db_store)
    # asyncio.run(sql_db_store.delete_table(table_name="meta_data"))
    asyncio.run(meta_manager.add(user_id=user_id, scope_id=scope_id))