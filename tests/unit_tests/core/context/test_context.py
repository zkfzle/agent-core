import unittest

from jiuwen.core.runtime.config import Config
from jiuwen.core.runtime.workflow import WorkflowRuntime, NodeRuntime
from jiuwen.core.runtime.state import ReadableStateLike
from jiuwen.core.runtime.workflow_state import InMemoryState
from jiuwen.core.runtime.utils import update_dict, get_by_schema


class ContextTest(unittest.TestCase):
    def assert_context(self, context: NodeRuntime, node_id: str, executable_id: str, parent_id: str):
        assert context.node_id() == node_id
        assert context.executable_id() == executable_id
        assert context.parent_id() == parent_id

    def test_basic(self):
        # Workflow context/
        context = WorkflowRuntime()
        context.state().commit_user_inputs({'a': 1, 'b': 2})
        assert context.state().get_global('a') == 1
        assert context.state().get_global('b') == 2

        # node1节点
        node1_context = NodeRuntime(context, "node1")
        assert node1_context.node_id() == "node1"
        assert node1_context.executable_id() == "node1"
        assert node1_context.parent_id() == ""
        assert node1_context.state().get_global('a') == 1
        assert node1_context.state().get_global('b') == 2
        # 通过input_schema获取inputs
        node1_input_schema = {"aa": "${a}", "bb": "${b}"}
        node1_input_schema2 = {"node_1_inputs": ["${a}", "${b}"]}
        assert node1_context.state().get_global(node1_input_schema) == {'aa': 1, 'bb': 2}
        assert node1_context.state().get_global(node1_input_schema2) == {"node_1_inputs": [1, 2]}

        # 通过transformer获取inputs
        def node1_transformer(state: ReadableStateLike):
            return state.get(node1_input_schema)

        assert node1_context.state().get_inputs_by_transformer(node1_transformer) == {'aa': 1, 'bb': 2}
        node1_context.state().update_global({"c": 3})
        node1_context.state().update({"url": "0.0.0.1"})
        node1_context.state().commit()
        assert node1_context.state().get_global('c') == 3
        assert node1_context.state().get('url') == '0.0.0.1'

        node2_context = NodeRuntime(context, "node2")
        assert node2_context.state().get_global('c') == 3
        assert node2_context.state().get('url') == None

        # 嵌套workflow
        sub_workflow_context = NodeRuntime(context, "sub_workflow1")
        sub_workflow_context.state().commit_user_inputs({'a': 11, 'b': 12})
        sub_workflow_context.state().commit()

        sub_node1_context = NodeRuntime(sub_workflow_context, "node1")
        assert sub_node1_context.node_id() == "node1"
        assert sub_node1_context.parent_id() == "sub_workflow1"
        assert sub_node1_context.executable_id() == "sub_workflow1.node1"
        assert sub_node1_context.state().get_global(node1_input_schema) == {'aa': 11, 'bb': 12}
        sub_node1_context.state().update_global({"c": 4})
        sub_node1_context.state().update({"url": "0.0.0.2"})
        sub_node1_context.state().commit()
        assert sub_node1_context.state().get_global('c') == 4
        assert sub_node1_context.state().get('url') == '0.0.0.2'

    def test_context_state(self):
        source = {}
        # 增加a.b: nums属性
        update_dict({"a.b.nums": [1, 2, 3]}, source)
        assert source == {'a': {'b': {'nums': [1, 2, 3]}}}
        # 增加a.b: name属性
        update_dict({
            "a.b.name": "shanghai"
        }, source)
        assert source == {'a': {'b': {'nums': [1, 2, 3], 'name': 'shanghai'}}}
        # 增加a.b: class属性
        update_dict({"a.b": {"class": "hha"}}, source)
        assert source == {'a': {'b': {'nums': [1, 2, 3], 'name': 'shanghai', 'class': 'hha'}}}
        # 覆盖a.b所有$ok
        update_dict({"a.b": [1, 2, 3]}, source)
        assert source == {'a': {'b': [1, 2, 3]}}
        assert get_by_schema("a", data=source) == {'b': [1, 2, 3]}
        assert get_by_schema({"a": "b"}, data=source) == {"a": "b"}
        assert get_by_schema({"result": "${a.b}"}, data=source) == {'result': [1, 2, 3]}
        assert get_by_schema({"result": ["abc", "${a}"]}, data=source) == {"result": ["abc", {'b': [1, 2, 3]}]}
        assert get_by_schema({"result": ["abc", "cde"]}, data=source) == {"result": ["abc", "cde"]}
        assert get_by_schema({"result": {"abc": "cde", "result": "${1}"}}, data=source) == {
            "result": {"abc": "cde", "result": None}}

        assert get_by_schema({"result": ["${abc}", "cde"]}, data=source) == {"result": [None, "cde"]}
        assert get_by_schema({"result": {"abc": "cde", "result": "${a}"}}, data=source) == {
            "result": {"abc": "cde", "result": {'b': [1, 2, 3]}}}

    def test_clean_non_value(self):
        data = {"a": {"a1": 1, "a2":2},  "b": {"b1": {"b11": "1",  "b12": [1,2,None], "b13": "2"}}, "c":2}
        update = {"c": None}
        update_dict(update, data)
        assert data == {"a": {"a1": 1, "a2":2},  "b": {"b1": {"b11": "1",  "b12": [1,2,None], "b13": "2"}}}
        update = {"a.a1": None}
        update_dict(update, data)
        assert data == {"a": {"a2":2},  "b": {"b1": {"b11": "1",  "b12": [1,2,None], "b13": "2"}}}
