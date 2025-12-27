import unittest
import asyncio
import json
from unittest.mock import AsyncMock
from openjiuwen.core.memory.search.search_manager.search_manager import SearchManager
from openjiuwen.core.memory.mem_unit.memory_unit import MemoryType
from openjiuwen.core.memory.manage.user_profile_manager import UserProfileManager
from openjiuwen.core.memory.manage.variable_manager import VariableManager
from openjiuwen.core.utils.llm.messages import AIMessage


class TestSearchManager(unittest.TestCase):

    def setUp(self):
        # 每个测试前的初始化
        pass

    def test_search_with_valid_search_type(self):
        # 测试使用有效搜索类型的情况
        mock_user_profile_manager = AsyncMock()
        mock_user_profile_manager.search.return_value = [
            {"id": "1", "mem": "I like apples", "score": 0.8},
            {"id": "2", "mem": "I live in Beijing", "score": 0.7},
            {"id": "3", "mem": "I live in Beijing", "score": 0.7},
            {"id": "4", "mem": "I live in Beijing", "score": 0.7}
        ]

        managers = {
            MemoryType.USER_PROFILE.value: mock_user_profile_manager
        }

        mock_user_mem_store = AsyncMock()
        search_manager = SearchManager(managers, mock_user_mem_store, '')

        result = asyncio.run(search_manager.search(
            user_id="user1",
            group_id="group1",
            query="What do I like?",
            top_k=2,
            search_type=MemoryType.USER_PROFILE.value
        ))

        self.assertEqual(len(result), 2)
        mock_user_profile_manager.search.assert_called_once()

    def test_search_with_invalid_search_type(self):
        # 测试使用无效搜索类型的情况
        managers = {}
        mock_user_mem_store = AsyncMock()
        search_manager = SearchManager(managers, mock_user_mem_store, '')

        with self.assertRaises(ValueError) as context:
            asyncio.run(search_manager.search(
                user_id="user1",
                group_id="group1",
                query="What do I like?",
                top_k=5,
                search_type="invalid_type"
            ))
        self.assertIn("invalid_type is not a valid search type", str(context.exception))

    def test_search_with_nonexistent_manager(self):
        # 测试搜索类型存在但对应的manager未初始化的情况
        managers = {}
        mock_user_mem_store = AsyncMock()
        search_manager = SearchManager(managers, mock_user_mem_store, '')

        with self.assertRaises(ValueError) as context:
            asyncio.run(search_manager.search(
                user_id="user1",
                group_id="group1",
                query="What do I like?",
                top_k=5,
                search_type=MemoryType.USER_PROFILE.value
            ))
        self.assertIn("memory manager not inited", str(context.exception))

    def test_search_without_search_type(self):
        # 测试不指定搜索类型的情况
        mock_user_profile_manager = AsyncMock()
        mock_user_profile_manager.search.return_value = [
            {"id": "1", "mem": "I like apples", "score": 0.8}
        ]

        mock_variable_manager = AsyncMock()
        mock_variable_manager.search.return_value = [
            {"id": "2", "mem": "age:25", "score": 0.9}
        ]

        managers = {
            MemoryType.USER_PROFILE.value: mock_user_profile_manager,
            MemoryType.VARIABLE.value: mock_variable_manager
        }

        mock_user_mem_store = AsyncMock()
        search_manager = SearchManager(managers, mock_user_mem_store, '')

        result = asyncio.run(search_manager.search(
            user_id="user1",
            group_id="group1",
            query="What do I like?",
            top_k=5
        ))

        # 只有user_profile类型会被搜索，因为它在user_mem_manager_list中
        self.assertEqual(len(result), 1)
        mock_user_profile_manager.search.assert_called_once()
        mock_variable_manager.search.assert_not_called()

    def test_list_user_mem(self):
        # 测试list_user_mem方法
        mock_user_mem_store = AsyncMock()
        mock_user_mem_store.get_in_range.return_value = [
            {"id": "1", "mem": "Message 1", "context_summary": "Summary1"},
            {"id": "2", "mem": "Message 2", "context_summary": "Summary2"}
        ]

        managers = {}
        search_manager = SearchManager(managers, mock_user_mem_store, '')

        result = asyncio.run(search_manager.list_user_mem(
            user_id="user1",
            group_id="group1",
            nums=2,
            pages=1
        ))

        self.assertEqual(len(result), 2)
        mock_user_mem_store.get_in_range.assert_called_once_with("user1", "group1", 0, 2)

    def test_list_user_profile_success(self):
        # 测试list_user_profile方法成功情况
        mock_user_profile_manager = AsyncMock(spec=UserProfileManager)
        mock_user_profile_manager.list_user_profile.return_value = [
            {"id": "1", "profile_type": "name", "profile_mem": "John"},
            {"id": "2", "profile_type": "location", "profile_mem": "New York"}
        ]

        managers = {
            MemoryType.USER_PROFILE.value: mock_user_profile_manager
        }

        mock_user_mem_store = AsyncMock()
        search_manager = SearchManager(managers, mock_user_mem_store, '')

        result = asyncio.run(search_manager.list_user_profile(
            user_id="user1",
            group_id="group1"
        ))

        self.assertEqual(len(result), 2)
        mock_user_profile_manager.list_user_profile.assert_called_once()

    def test_list_user_profile_with_profile_type(self):
        # 测试list_user_profile方法带profile_type参数的情况
        mock_user_profile_manager = AsyncMock(spec=UserProfileManager)
        mock_user_profile_manager.list_user_profile.return_value = [
            {"id": "1", "profile_type": "name", "profile_mem": "John"}
        ]

        managers = {
            MemoryType.USER_PROFILE.value: mock_user_profile_manager
        }

        mock_user_mem_store = AsyncMock()
        search_manager = SearchManager(managers, mock_user_mem_store, '')

        result = asyncio.run(search_manager.list_user_profile(
            user_id="user1",
            group_id="group1",
            profile_type="name"
        ))

        self.assertEqual(len(result), 1)
        mock_user_profile_manager.list_user_profile.assert_called_once_with(
            user_id="user1",
            group_id="group1",
            profile_type="name"
        )

    def test_list_user_profile_wrong_manager_type(self):
        # 测试list_user_profile方法使用错误的manager类型的情况
        mock_variable_manager = AsyncMock(spec=VariableManager)

        managers = {
            MemoryType.USER_PROFILE.value: mock_variable_manager
        }

        mock_user_mem_store = AsyncMock()
        search_manager = SearchManager(managers, mock_user_mem_store, '')

        with self.assertRaises(ValueError) as context:
            asyncio.run(search_manager.list_user_profile(
                user_id="user1",
                group_id="group1"
            ))
        self.assertIn("manager class is not UserProfileManager", str(context.exception))

    def test_get_user_variable(self):
        # 测试get_user_variable方法
        mock_variable_manager = AsyncMock(spec=VariableManager)
        mock_variable_manager.query_variable.return_value = {"age": "25"}

        managers = {
            MemoryType.VARIABLE.value: mock_variable_manager
        }

        mock_user_mem_store = AsyncMock()
        search_manager = SearchManager(managers, mock_user_mem_store, '')

        result = asyncio.run(search_manager.get_user_variable(
            user_id="user1",
            group_id="group1",
            var_name="age"
        ))

        self.assertEqual(result, "25")
        mock_variable_manager.query_variable.assert_called_once()

    def test_get_all_user_variable(self):
        # 测试get_all_user_variable方法
        mock_variable_manager = AsyncMock(spec=VariableManager)
        mock_variable_manager.query_variable.return_value = {"age": "25", "name": "John"}

        managers = {
            MemoryType.VARIABLE.value: mock_variable_manager
        }

        mock_user_mem_store = AsyncMock()
        search_manager = SearchManager(managers, mock_user_mem_store, '')

        result = asyncio.run(search_manager.get_all_user_variable(
            user_id="user1",
            group_id="group1"
        ))

        self.assertEqual(result, {"age": "25", "name": "John"})
        mock_variable_manager.query_variable.assert_called_once()


    def test_list_user_mem_with_error(self):
        mock_user_mem_store = AsyncMock()
        mock_user_mem_store.get_in_range.return_value = None
        managers = {}
        search_manager = SearchManager(managers, mock_user_mem_store, '')

        result = asyncio.run(search_manager.list_user_mem(
            user_id="user1",
            group_id="group1",
            nums=-1,
            pages=-1
        ))
        self.assertEqual(result, None)


    def test_decompose_query_success(self):
        search_manager = SearchManager({}, None, "")

        mock_model_client = AsyncMock()
        mock_model_client.ainvoke.return_value = AIMessage(content=json.dumps(["sub query 1", "sub query 2"]))

        base_chat_model = ("mock-model", mock_model_client)

        result = asyncio.run(
            search_manager.decompose_query(
                base_chat_model=base_chat_model,
                query="original query"
            )
        )

        self.assertEqual(result, ["sub query 1", "sub query 2"])
        mock_model_client.ainvoke.assert_called_once()

    def test_decompose_query_invalid_result(self):
        search_manager = SearchManager({}, None, "")

        mock_model_client = AsyncMock()
        mock_model_client.ainvoke.return_value.content = json.dumps({"key": "value"})
        base_chat_model = ("mock-model", mock_model_client)

        result = asyncio.run(
            search_manager.decompose_query(
                base_chat_model=base_chat_model,
                query="original query"
            )
        )

        self.assertEqual(result, [])

    def test_rewrite_and_search_success(self):
        search_manager = SearchManager({}, None, "")

        # mock decompose_query
        search_manager.decompose_query = AsyncMock(
            return_value=["q1", "q2"]
        )

        # mock search
        search_manager.search = AsyncMock(side_effect=[
            [
                {"id": "1", "score": 0.9},
                {"id": "2", "score": 0.8},
            ],
            [
                {"id": "3", "score": 0.95},
                {"id": "4", "score": 0.7},
            ]
        ])

        result = asyncio.run(
            search_manager.rewrite_and_search(
                base_chat_model=("model", AsyncMock()),
                user_id="u1",
                group_id="g1",
                query="complex query",
                top_k=4,
                threshold=0.75
            )
        )

        # 只保留 score >= threshold
        self.assertEqual(len(result), 3)
        self.assertEqual(
            [r["id"] for r in result],
            ["3", "1", "2"]  # 按 score 排序
        )

        self.assertEqual(search_manager.search.call_count, 2)

    def test_rewrite_and_search_fallback_query(self):
        search_manager = SearchManager({}, None, "")

        search_manager.decompose_query = AsyncMock(return_value=[])
        search_manager.search = AsyncMock(return_value=[
            {"id": "1", "score": 0.9},
            {"id": "2", "score": 0.2}
        ])

        result = asyncio.run(
            search_manager.rewrite_and_search(
                base_chat_model=("model", AsyncMock()),
                user_id="u1",
                group_id="g1",
                query="original query",
                top_k=2,
                threshold=0.3
            )
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "1")

        search_manager.search.assert_called_once()


if __name__ == '__main__':
    unittest.main()