import unittest

from jiuwen.core.context.config import Config
from jiuwen.core.context.context import WorkflowContext, NodeContext
from jiuwen.core.context.state import InMemoryState, ReadableStateLike


class ContextTest(unittest.TestCase):
    def assert_context(self, context: NodeContext, node_id: str, executable_id: str, parent_id: str):
        assert context.node_id() == node_id
        assert context.executable_id() == executable_id
        assert context.parent_id() == parent_id

    def test_basic(self):
        # Workflow context/
        context = WorkflowContext(config=Config(), state=InMemoryState(), store=None)
        context.state().commit_user_inputs({'a': 1, 'b': 2})
        assert context.state().get_global('a') == 1
        assert context.state().get_global('b') == 2

        # node1节点
        node1_context = NodeContext(context, "node1")
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

        node2_context = NodeContext(context, "node2")
        assert node2_context.state().get_global('c') == 3
        assert node2_context.state().get('url') == None

        # 嵌套workflow
        sub_workflow_context = NodeContext(context, "sub_workflow1")
        sub_workflow_context.state().commit_user_inputs({'a': 11, 'b': 12})
        sub_workflow_context.state().commit()

        sub_node1_context = NodeContext(sub_workflow_context, "node1")
        assert sub_node1_context.node_id() == "node1"
        assert sub_node1_context.parent_id() == "sub_workflow1"
        assert sub_node1_context.executable_id() == "sub_workflow1.node1"
        assert sub_node1_context.state().get_global(node1_input_schema) == {'aa': 11, 'bb': 12}
        sub_node1_context.state().update_global({"c": 4})
        sub_node1_context.state().update({"url": "0.0.0.2"})
        sub_node1_context.state().commit()
        assert sub_node1_context.state().get_global('c') == 4
        assert sub_node1_context.state().get('url') == '0.0.0.2'
