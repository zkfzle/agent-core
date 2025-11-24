#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import Any, Dict, List, Tuple
from sqlalchemy import (text, engine, insert, update,
                        select, delete, exists, Table, MetaData,
                        Column, Text, column, and_, or_, desc, asc)
import logging

logger = logging.getLogger(__name__)

class SqlDbStore():
    def __init__(self, conn_pool: engine.Engine):
        self.conn_pool = conn_pool
        self._table_cache: dict[str, Table] = {}

    def _get_table(self, table_name: str) -> Table:
        if table_name in self._table_cache:
            return self._table_cache[table_name]
        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=self.conn_pool)
        self._table_cache[table_name] = table
        return table

    def write(self, table: str, data: dict) -> bool:
        t = self._get_table(table)
        stmt = insert(t).values(**data)
        try:
            with self.conn_pool.begin() as conn:
                conn.execute(stmt)
            return True
        except Exception as e:
            logger.error("Write failed", exc_info=e)
            return False

    def get(self, table: str, id: str, columns: list[str] = []) -> dict[str, Any] | None:
        try:
            t = self._get_table(table)
            if columns:
                cols = [t.c[col] for col in columns]
                stmt = select(*cols)
            else:
                stmt = select(t)
            stmt = stmt.where(t.c.id == id)
            with self.conn_pool.connect() as conn:
                row = conn.execute(stmt).mappings().first()
                return dict(row) if row else None
        except Exception as e:
            logger.error("Failed to get data", exc_info=e)
            return None

    def get_with_sort(self, table: str, filters: Dict[str, Any], sort_by: str = "timestamp",
                      order: str = "DESC", limit: int = 100) -> List[Dict[str, Any]]:
        try:
            t = self._get_table(table)
            if sort_by not in t.c:
                raise ValueError(f"Sort column '{sort_by}' does not exist in table '{table}'")
            clauses = [
                t.c[col] == val for col, val in filters.items() if col in t.c
            ]
            stmt = select(t)
            if clauses:
                stmt = stmt.where(and_(*clauses))
            if order.upper() == "DESC":
                stmt = stmt.order_by(desc(t.c[sort_by]))
            else:
                stmt = stmt.order_by(asc(t.c[sort_by]))
            stmt = stmt.limit(limit)
            with self.conn_pool.connect() as conn:
                result = conn.execute(stmt).mappings().fetchall()
                return [dict(row) for row in result]
        except Exception as e:
            logger.error("Failed to fetch filtered and sorted data", exc_info=e)
            return []

    def exist(self, table: str, conditions: Dict[str, Any]) -> bool:
        t = self._get_table(table)
        clauses = [t.c[col] == val for col, val in conditions.items()]
        stmt = select(1).where(and_(*clauses))
        with self.conn_pool.connect() as conn:
            return conn.execute(stmt).first() is not None

    def batch_get(self, table: str, conditions_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        t = self._get_table(table)
        clauses = [or_(*[t.c[col] == val for col, val in cond.items()]) for cond in conditions_list]
        stmt = select(t).where(or_(*clauses)) if clauses else select(t)
        with self.conn_pool.connect() as conn:
            return [dict(r) for r in conn.execute(stmt).mappings().fetchall()]

    def condition_get(self, table: str, conditions: Dict[str, List[Any]],
                      columns: List[str] = []) -> List[Dict[str, Any]] | None:
        try:
            t: Table = self._get_table(table)
            stmt = (
                select(t) if not columns
                else select(*[t.c[col] for col in columns])
            )
            clause_list = []
            for col, values in conditions.items():
                if not isinstance(values, list):
                    raise TypeError(f"condition[{col}] must be a List")
                clause_list.append(t.c[col].in_(values))
            if clause_list:
                stmt = stmt.where(and_(*clause_list))
            with self.conn_pool.connect() as conn:
                rows = conn.execute(stmt).mappings().fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error("Failed to get data via condition_get", exc_info=e)
            return None

    def update(self, table: str, conditions: dict, data: dict) -> bool:
        t = self._get_table(table)
        clauses = [t.c[col].in_(vals) if isinstance(vals, list) else t.c[col] == vals
                   for col, vals in conditions.items()]
        stmt = update(t).where(and_(*clauses)).values(**data)
        try:
            with self.conn_pool.begin() as conn:
                conn.execute(stmt)
            return True
        except Exception as e:
            logger.error("Update failed", exc_info=e)
            return False

    def delete(self, table: str, conditions: dict) -> bool:
        t = self._get_table(table)
        clauses = [t.c[col].in_(vals) if isinstance(vals, list) else t.c[col] == vals
                   for col, vals in conditions.items()]
        stmt = delete(t).where(and_(*clauses))
        try:
            with self.conn_pool.begin() as conn:
                conn.execute(stmt)
            return True
        except Exception as e:
            logger.error("Delete failed", exc_info=e)
            return False

    def delete_table(self, table_name: str) -> bool:
        try:
            metadata = MetaData()
            t = Table(table_name, metadata)
            t.drop(self.conn_pool, checkfirst=True)
            return True
        except Exception as e:
            logger.error("Delete table failed", exc_info=e)
            return False
