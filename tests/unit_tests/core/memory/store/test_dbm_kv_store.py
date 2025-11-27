import os
import unittest
from openjiuwen.core.memory.store.impl.dbm_kv_store import DbmKVStore   # ← 修改为你的真实模块路径


class TestDBMStore(unittest.TestCase):

    def setUp(self):
        # 测试前创建临时目录
        self.test_dir = "test_dbm"
        os.makedirs(self.test_dir, exist_ok=True)
        self.db_path = os.path.join(self.test_dir, "testdb")
        self.store = DbmKVStore(self.db_path, cache_size=4)

    def test_set_and_get(self):
        self.store.set("a", "123")
        self.assertEqual(self.store.get("a"), "123")

    def test_exists(self):
        self.store.set("b", "hello")
        self.assertTrue(self.store.exists("b"))
        self.assertFalse(self.store.exists("xxx"))

    def test_delete(self):
        self.store.set("c", "delme")
        self.assertTrue(self.store.exists("c"))
        self.store.delete("c")
        self.assertFalse(self.store.exists("c"))
        self.assertIsNone(self.store.get("c"))

    def test_mget(self):
        self.store.set("k1", "v1")
        self.store.set("k2", "v2")
        res = self.store.mget(["k1", "k2", "k3"], default="")
        self.assertEqual(res, ["v1", "v2", ""])

    def test_lru_cache_is_cleared(self):
        self.store.set("x", "1")
        v1 = self.store.get("x")  # cached
        self.store.set("x", "2")  # should clear cache
        v2 = self.store.get("x")
        self.assertEqual(v2, "2")

    def test_db_files_created(self):
        self.store.set("a", "1")
        self.store.close()
        files = os.listdir(self.test_dir)
        self.assertTrue(len(files) > 0, "dbm files should be created")

    def test_get_by_regex(self):
        self.store.set("session_summary\x1Fuser1\x1Fapp1\x1Fsession1", "mock_value")
        self.store.set("session_summary\x1Fuser1\x1Fapp2\x1Fsession2", "mock_value")
        self.store.set("session_summary\x1Fuser1\x1Fapp1\x1Fsession3", "mock_value")

        res = self.store.get_by_regex("session_summary\x1Fuser1\x1F.*$")
        self.assertEqual(res, {
            "session_summary\x1Fuser1\x1Fapp1\x1Fsession1": "mock_value",
            "session_summary\x1Fuser1\x1Fapp2\x1Fsession2": "mock_value",
            "session_summary\x1Fuser1\x1Fapp1\x1Fsession3": "mock_value"
        })

    def test_delete_by_regex(self):
        self.store.set("session_summary\x1Fuser1\x1Fapp1\x1Fsession1", "mock_value")
        self.store.set("session_summary\x1Fuser1\x1Fapp2\x1Fsession2", "mock_value")
        self.store.set("session_summary\x1Fuser1\x1Fapp1\x1Fsession3", "mock_value")
        self.store.set("session_summary\x1Fuser2\x1Fapp1\x1Fsession4", "mock_value")

        self.store.delete_by_regex("^session_summary\x1F+user1+\x1F.*$")
        self.assertFalse(self.store.exists("session_summary\x1Fuser1\x1Fapp1\x1Fsession1"))
        self.assertFalse(self.store.exists("session_summary\x1Fuser1\x1Fapp2\x1Fsession2"))
        self.assertFalse(self.store.exists("session_summary\x1Fuser1\x1Fapp1\x1Fsession3"))
        self.assertTrue(self.store.exists("session_summary\x1Fuser2\x1Fapp1\x1Fsession4"))

    def test_delete_by_regex_1(self):
        self.store.set("session_summary\x1Fuser1\x1Fapp1\x1Fsession1", "mock_value")
        self.store.set("session_summary\x1Fuser1\x1Fapp2\x1Fsession2", "mock_value")
        self.store.set("session_summary\x1Fuser1\x1Fapp1\x1Fsession3", "mock_value")
        self.store.set("session_summary\x1Fuser2\x1Fapp1\x1Fsession4", "mock_value")

        self.store.delete_by_regex("^[^\x1F]+\x1Fuser1\x1F.*$")
        self.assertFalse(self.store.exists("session_summary\x1Fuser1\x1Fapp1\x1Fsession1"))
        self.assertFalse(self.store.exists("session_summary\x1Fuser1\x1Fapp2\x1Fsession2"))
        self.assertTrue(self.store.exists("session_summary\x1Fuser2\x1Fapp1\x1Fsession4"))

if __name__ == "__main__":
    unittest.main()
