import random
from datetime import datetime
from openjiuwen.memory.store.user_mem_store import UserMemStore
import unittest
from openjiuwen.memory.store.dbm_kv_store import DbmKVStore
import os


class TestUserMemStore(unittest.TestCase):
    @staticmethod
    def _generate_next_id() -> str:
        return str(random.randint(0, 2 ** 31 - 1))

    def test_basic(self):
        # kv_store_instance = SqliteKVStore(".")
        test_dir = "test_dbm"
        os.makedirs(test_dir, exist_ok=True)
        db_path = os.path.join(test_dir, "testdb")
        kv_store_instance = DbmKVStore(db_path)

        store = UserMemStore(kv_store_instance=kv_store_instance)

        user_profile_mem_type = "user_profile"
        episodic_mem_type = "episodic_mem"

        # Test write and get
        user_id1 = "user1"
        app_id1 = "app1"
        mem_id1 = TestUserMemStore._generate_next_id()
        profile_type1 = "personal_information"
        data1 = {
            "id": mem_id1,
            "user_id": user_id1,
            "app_id": app_id1,
            "profile_type": profile_type1,
            "profile_mem": "user profile1",
            "mem_type": user_profile_mem_type,
            "time": str(datetime.now()),
        }
        self.assertTrue(store.write(user_id1, app_id1, mem_id1, data1))
        user_profile_data1 = store.get(user_id1, app_id1, mem_id1)
        self.assertEqual(user_profile_data1, data1)

        mem_id2 = TestUserMemStore._generate_next_id()
        data2 = {
            "id": mem_id2,
            "user_id": user_id1,
            "app_id": app_id1,
            "content": "episodic memory 1",
            "mem_type": episodic_mem_type,
            "time": str(datetime.now()),
        }
        self.assertTrue(store.write(user_id1, app_id1, mem_id2, data2))

        user_id2 = "user2"
        app_id2 = "app2"
        mem_id3 = TestUserMemStore._generate_next_id()
        data3 = {
            "id": mem_id3,
            "user_id": user_id2,
            "app_id": app_id2,
            "content": "episodic memory 2",
            "mem_type": episodic_mem_type,
            "time": str(datetime.now()),
        }
        self.assertTrue(store.write(user_id2, app_id2, mem_id3, data3))

        # Test update and get
        modify_user_profile_mem = "user profile 2"
        self.assertTrue(store.update(user_id1, app_id1, mem_id1, {"profile_mem": modify_user_profile_mem}))
        user_profile_update_data1 = store.get(user_id1, app_id1, mem_id1)
        self.assertEqual(user_profile_update_data1.get("profile_mem"), modify_user_profile_mem)

        # Test get_all
        all_user1_data_list = store.get_all(user_id1, app_id1)
        self.assertEqual(len(all_user1_data_list), 2)
        # get all user profile data
        all_user1_profile_list = store.get_all(user_id1, app_id1, user_profile_mem_type)
        self.assertEqual(len(all_user1_profile_list), 1)
        # get all episodic mem data
        all_user1_episodic_list = store.get_all(user_id1, app_id1, episodic_mem_type)
        self.assertEqual(len(all_user1_episodic_list), 1)

        all_user2_data_list = store.get_all(user_id2, app_id2)
        self.assertEqual(len(all_user2_data_list), 1)

        # Test batch_get
        batch_get_user1_data_list = store.batch_get(user_id1, app_id1, [mem_id1, mem_id2])
        self.assertEqual(len(batch_get_user1_data_list), 2)
        self.assertNotEqual(batch_get_user1_data_list[0], None)
        self.assertNotEqual(batch_get_user1_data_list[1], None)

        # Test get_by_id
        get_data2 = store.get_by_id(mem_id2)
        self.assertEqual(data2, get_data2)

        # Test get_by_topic
        topic_data = store.get_by_topic(user_id1, app_id1, profile_type1)
        self.assertEqual(len(topic_data), 1)

        # Test delete and delete_by_user
        self.assertTrue(store.delete_be_user(user_id1, app_id1))
        self.assertTrue(store.delete(user_id2, app_id2, mem_id3))
        self.assertEqual(store.get_all(user_id1, app_id1), None)
        self.assertEqual(store.get_all(user_id2, app_id2), None)

if __name__ == "__main__":
    testIns = TestUserMemStore()
    testIns.test_basic()
