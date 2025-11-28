import unittest
from enum import StrEnum
from sqlalchemy import engine, text, create_engine, QueuePool
from pathlib import Path
import os

from openjiuwen.core.memory.store.base_semantic_store import SearchHit
os.environ['HF_ENDPOINT']= "https://hf-mirror.com"
from openjiuwen.core.memory.manage.data_id_manager import DataIdManager
from openjiuwen.core.memory.manage.message_manager import MessageManager
from openjiuwen.core.memory.manage.user_profile_manager import UserProfileManager
from openjiuwen.core.memory.manage.variable_manager import VariableManager
from openjiuwen.core.memory.manage.write_manager import WriteManager
from openjiuwen.core.memory.mem_unit.memory_unit import UserProfileUnit, VariableUnit, MemoryType, ConflictType
from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.store.user_mem_store import UserMemStore
from openjiuwen.core.memory.store.sql_db_store import SqlDbStore
from openjiuwen.memory.store.dbm_kv_store import DbmKVStore as MockKVStore
from openjiuwen.core.memory.config.config import Config


class ContextStoreColumnType(StrEnum):
    TEXT = 'TEXT'
    INTEGER = 'INTEGER'
    REAL = 'REAL'
    BLOB = 'BLOB'
    NUMERIC = 'NUMERIC'


CONTEXT_CONFIG = {
    'table': 'user_message',
    'columns': {
        'message_id': ContextStoreColumnType.TEXT,
        'user_id': ContextStoreColumnType.TEXT,
        'session_id': ContextStoreColumnType.TEXT,
        'app_id': ContextStoreColumnType.TEXT,
        'role': ContextStoreColumnType.TEXT,
        'content': ContextStoreColumnType.TEXT,
        'timestamp': ContextStoreColumnType.TEXT,
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
                alter_sql = f"ALTER TABLE {table} ADD COLUMN '{column_name}' {column_type}"
                conn.execute(text(alter_sql))
            conn.commit()
    except Exception as e:
        logger.error("Failed to create table", exc_info=e)

config = Config(
    variables_key={"key":["value"]},
    model_api_base="http://test.com",
    model_api_key="test_key",
    model_name="test_model",
    model_provider="test_provider",
    strategy=["test_strategy"],
    vector_store_dir=".",
    kv_store_dir="."
)

# Mock语义存储实现，避免实际模型加载
class MockSemanticStore:
    """Mock语义存储，用于测试环境，不依赖实际模型"""
    
    def __init__(self, config, model_config):
        self.memory_store = {}
        self.config = config
        self.model_config = model_config
    
    def add(self, mem, memory_id, user_id, app_id, mem_type=None):
        """模拟添加记忆"""
        index_name = f"{user_id}^{app_id}^{mem_type or 'default'}"
        if index_name not in self.memory_store:
            self.memory_store[index_name] = {}
        
        for m, mid in zip(mem, memory_id):
            self.memory_store[index_name][mid] = {
                'content': m,
                'user_id': user_id,
                'app_id': app_id,
                'mem_type': mem_type
            }
    
    def remove(self, ids, user_id, app_id, mem_type=None):
        """模拟删除记忆"""
        index_name = f"{user_id}^{app_id}^{mem_type or 'default'}"
        if index_name in self.memory_store:
            for id_to_remove in ids:
                self.memory_store[index_name].pop(id_to_remove, None)
    
    def search(self, query, user_id, app_id, mem_type=None, top_k=5):
        """模拟搜索功能，返回匹配的记忆"""
        index_name = f"{user_id}^{app_id}^{mem_type or 'default'}"
        if index_name not in self.memory_store:
            return []
        
        # 简单的文本匹配搜索
        results = []
        for memory_id, memory_data in self.memory_store[index_name].items():
            content = memory_data['content']
            # 简单的关键词匹配
            if any(q in content for q in query):
                # 模拟返回SearchHit对象
                results.append(SearchHit(id=memory_id, distance=0.0))
        
        # 返回top_k个结果
        return results
    
    def get_memory(self, memory_id):
        """获取特定记忆"""
        for index_name, memories in self.memory_store.items():
            if memory_id in memories:
                return memories[memory_id]
        return None
    
    def update_memory(self, memory_id, new_content):
        """更新记忆内容"""
        for index_name, memories in self.memory_store.items():
            if memory_id in memories:
                memories[memory_id]['content'] = new_content
                return True
        return False
    
    def list_memories(self, user_id, app_id, mem_type=None):
        """列出用户的所有记忆"""
        index_name = f"{user_id}^{app_id}^{mem_type or 'default'}"
        if index_name not in self.memory_store:
            return []
        
        return [{'id': k, 'content': v['content']} for k, v in self.memory_store[index_name].items()]
    
    def delete_index_by_match(self, match_str):
        """模拟删除索引功能"""
        # 解析匹配字符串，格式为 user_id^app_id^mem_type^suffix
        parts = match_str.split('^')
        if len(parts) != 4:
            logger.error(f"Invalid match_str: {match_str}")
            return
        
        # 根据匹配模式删除对应的索引
        user_pattern, app_pattern, type_pattern, suffix_pattern = parts
        to_delete = []
        
        for index_name in self.memory_store:
            index_parts = index_name.split('^')
            if len(index_parts) >= 3:  # user_id^app_id^mem_type
                idx_user, idx_app, idx_type = index_parts[0], index_parts[1], index_parts[2]
                
                # 检查是否匹配
                user_match = (user_pattern == "*" or user_pattern == idx_user)
                app_match = (app_pattern == "*" or app_pattern == idx_app)
                type_match = (type_pattern == "*" or type_pattern == idx_type)
                
                if user_match and app_match and type_match:
                    to_delete.append(index_name)
        
        # 删除匹配的索引
        for index_name in to_delete:
            del self.memory_store[index_name]

class TestManage(unittest.TestCase):
    def test_basic(self):
        mock_kv_store = MockKVStore("kv_db")
        data_id_generator = DataIdManager(mock_kv_store)
        
        # 使用Mock语义存储替代实际模型
        mock_semantic_recall = MockSemanticStore(config, None)
        
        # path = Path("./sql_db.db")
        # conn = create_engine(
        #     f"sqlite:///{path.resolve()}",
        #     poolclass=QueuePool,
        #     pool_size=10,
        #     max_overflow=20,
        #     pool_pre_ping=True,
        #     pool_recycle=3600
        # )
        # create(conn, CONTEXT_CONFIG['table'], CONTEXT_CONFIG['columns'])
        # mock_db_store = SqlDbStore(conn)
        # message_manager = MessageManager(mock_db_store, data_id_generator)
        mock_mem_store = UserMemStore(mock_kv_store)
        user_profile_manager = UserProfileManager(
            semantic_recall_instance=mock_semantic_recall, 
            user_mem_store=mock_mem_store,
            data_id_generator=data_id_generator
        )
        variable_manager = VariableManager(mock_kv_store)
        managers = {"user_profile": user_profile_manager, "variable": variable_manager}
        write_manager = WriteManager(managers, mock_mem_store)
        test_all_data = [
            {"user_id": "usrZH2025", "app_id": "fitnesstrackerv3", "profile_type": "interests_hobbies",
             "profile_mem": "用户非常喜欢川菜，尤其是水煮鱼和麻婆豆腐"},
            {"user_id": "usrZH2025", "app_id": "fitnesstrackerv3", "profile_type": "personal_information",
             "profile_mem": "用户的职业是软件工程师，居住在北京市"},
            {"user_id": "usrZH2025", "app_id": "fitnesstrackerv3", "profile_type": "personal_information",
             "profile_mem": "用户的副业是抖音直播"},
            {"user_id": "usrZH2025", "app_id": "fitnesstrackerv3", "profile_type": "assert_information",
             "profile_mem": "用户的银行账户余额为10000元"},
            {"user_id": "usrZH2025", "app_id": "fitnesstrackerv3", "profile_type": "social_information",
             "profile_mem": "用户的朋友圈中有50个好友"},
            {"user_id": "usrZH2025", "app_id": "fitnesstrackerv3", "profile_type": "other_information",
             "profile_mem": "用户的宠物是一只金毛犬"},
            {"user_id": "usrZH2026", "app_id": "fitnesstrackerv3", "profile_type": "interests_hobbies",
             "profile_mem": "用户喜欢打篮球和阅读历史小说"},
            {"user_id": "usrZH2026", "app_id": "fitnesstrackerv3", "profile_type": "personal_information",
             "profile_mem": "用户的生日是1990年1月1日"},
            {"user_id": "usrZH2026", "app_id": "fitnesstrackerv3", "profile_type": "assert_information",
             "profile_mem": "用户的汽车型号是特斯拉Model 3"},
            {"user_id": "usrZH2026", "app_id": "fitnesstrackerv3", "profile_type": "interests_hobbies",
             "profile_mem": "用户在Twitter上有200个关注者"},
        ]

        for item in test_all_data:
            conflict_info = {'id': '-1', "event": ConflictType.ADD.value, "text": item["profile_mem"]}
            mem_unit = UserProfileUnit(mem_type=MemoryType.USER_PROFILE, conflict_info=[conflict_info], **item)
            write_manager.add_mem([mem_unit])
            mem_unit = VariableUnit(mem_type=MemoryType.VARIABLE, variable_name=item['profile_type'],
                                    variable_mem=item['profile_mem'], user_id=item['user_id'], app_id=item['app_id'])
            write_manager.add_mem([mem_unit])
            # message_manager.add(user_id=item['user_id'], app_id=item['app_id'], role='user', content=item['profile_mem'])

        # message = message_manager.get(user_id=test_all_data[0]['user_id'], app_id=test_all_data[0]['app_id'], message_len=3)
        query = "用户的职业"
        res = variable_manager.query_variable(user_id=test_all_data[0]['user_id'], app_id=test_all_data[0]['app_id'])
        res = user_profile_manager.search(query, 5, user_id="usrZH2025", app_id="fitnesstrackerv3")
        self.assertEqual(1, len(res))
        # message_by_id = message_manager.get_by_id("15")

        user_profile_manager.update(res[0]['id'], "用户不是软件工程师，是系统")
        self.assertEqual("用户不是软件工程师，是系统", user_profile_manager.get(res[0]['id'])['mem'])

        res = user_profile_manager.list_user_profile("usrZH2025", "fitnesstrackerv3")
        self.assertEqual(6, len(res))

        res = user_profile_manager.list_user_profile("usrZH2025", "fitnesstrackerv3", "personal_information")
        self.assertEqual(2, len(res))
        for rr in res:
            write_manager.delete_mem_by_id(rr['id'])

        res = user_profile_manager.search(query, 5, user_id="userZH2025", app_id="fitnesstrackerv3")
        self.assertEqual(0, len(res))
        write_manager.delete_mem_by_user_id("userZH2026", "fitnesstrackerv3")
        res = user_profile_manager.search(query, 5, user_id="userZH2026", app_id="fitnesstrackerv3")
        self.assertEqual(0, len(res))

if __name__ == '__main__':
    unittest.main()