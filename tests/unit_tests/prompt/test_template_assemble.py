import unittest

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.utils.llm.messages import BaseMessage
from jiuwen.core.utils.prompt.assemble.variables.variable import Variable
from jiuwen.core.utils.prompt.template.template import Assembler
from jiuwen.core.utils.prompt.assemble.variables.textable import TextableVariable
from jiuwen.core.utils.prompt.index.template_store.template_store import Template


class TestPromptAssemble(unittest.TestCase):
    def test_textable_variable(self):
        self.assertRaises(JiuWenBaseException, TextableVariable, text="{{}}")
        var1 = TextableVariable(text="{{x}}")
        self.assertEqual(["x"], var1.input_keys)
        self.assertEqual("default", var1.name)

        var2 = TextableVariable(text="{{x}}{{y}}")
        self.assertEqual(["x", "y"], var2.input_keys)
        self.assertEqual("12", var2.eval(x="1", y="2"))
        self.assertEqual("12", var2.value)
        self.assertRaises(JiuWenBaseException, var2.eval, x=1, y=2, z=3)

    def test_textable_variables(self):
        self.assertRaises(JiuWenBaseException, TextableVariable, text="{{}}")
        var1 = TextableVariable(text="{{x}}")
        self.assertEqual(["x"], var1.input_keys)
        self.assertEqual("default", var1.name)

        var2 = TextableVariable(text="{{x}}{{y}}")
        self.assertEqual({"x", "y"}, set(var2.input_keys))
        self.assertRaises(JiuWenBaseException, var2.eval, x=1, y=2, z=3)
        self.assertEqual("12", var2.eval(x="1", y="2"))
        self.assertEqual("12", var2.value)

    def test_initialization(self):
        text = "You're an expert in the domain of {{domain}}"
        var = TextableVariable(text=text, name="role")
        self.assertEqual(text, var.text)
        self.assertEqual("role", var.name)
        self.assertEqual(["domain"], var.input_keys)
        self.assertEqual(["domain"], var.placeholders)

        text = "Hello, {{user.name}}"
        var = TextableVariable(text=text)
        self.assertEqual(["user"], var.input_keys)
        self.assertEqual(["user.name"], var.placeholders)

        text = "Hello, {{}}!"
        with self.assertRaises(JiuWenBaseException):
            TextableVariable(text=text)

    def test_update(self):
        text = "You're an expert in the domain of {{domain}}."
        var = TextableVariable(text=text)
        var.update(domain="science")
        self.assertEqual("You're an expert in the domain of science.", var.value)

        text = "This value is {{value}}."
        var = TextableVariable(text=text)
        var.update(value=42)
        self.assertEqual("This value is 42.", var.value)

    def test_eval(self):
        text = "You're an expert in the domain of {{domain}}."
        var = TextableVariable(text=text)
        result = var.eval(domain="science")
        self.assertEqual("You're an expert in the domain of science.", result)

        text = "Hello, {{user.name}}!"
        var = TextableVariable(text=text)
        result = var.eval(user={"name": "Alice"})
        self.assertEqual("Hello, Alice!", result)

        text = "Hello, {{name}}!"
        var = TextableVariable(text=text)
        with self.assertRaises(JiuWenBaseException):
            var.eval(wrong_key="Alice")

    def test_variable_initialization(self):
        var = Variable(name="test_var", input_keys=["key1", "key2"])
        self.assertEqual("test_var", var.name)
        self.assertEqual(["key1", "key2"], var.input_keys)
        self.assertEqual(var.value, "")

        var = Variable(name="test_var", input_keys=None)
        self.assertIsNone(var.input_keys)

    def test_prepare_inputs(self):
        var = Variable(name="test_var", input_keys=["key1", "key2"])

        input_kwargs = var._prepare_inputs(key1="value1", key2="value2")
        self.assertEqual({"key1": "value1", "key2": "value2"}, input_kwargs)

        with self.assertRaises(JiuWenBaseException):
            var._prepare_inputs(key1="value1")

        with self.assertRaises(JiuWenBaseException):
            var._prepare_inputs(key1="value1", key2="value2", key3="value3")

    def test_variable_eval(self):
        class MockVariable(Variable):
            def update(self, **kwargs):
                self.value = kwargs.get("key1", "") + kwargs.get("key2", "")

        var = MockVariable(name="test_var", input_keys=["key1", "key2"])

        result = var.eval(key1="value1", key2="value2")
        self.assertEqual("value1value2", result)

        with self.assertRaises(JiuWenBaseException):
            var.eval(key1="value1")

        with self.assertRaises(JiuWenBaseException):
            var.eval(key1="value1", key2="value2", key3="value3")

    def test_assemble(self):
        asm1 = Assembler(
            template_content="`#system#`{{role}}`#user#`{{memory}}",
            role=TextableVariable(text="你是一个精通{{domain}}领域的问答助手。")
        )
        self.assertEqual({"domain", "memory"}, set(asm1.input_keys))
        self.assertIsInstance(asm1.assemble(memory=[{"role": "user", "content": "我是谁"}], domain="科学"), list)

        asm2 = Assembler(
            template_content="`#assistant#`消息内容`*function_call*`{\"name\":\"func1\", \"arguments\":\"x\"}"
        )
        self.assertEqual([
            {"role": "assistant", "content": "消息内容", "function_call": {"name": "func1", "arguments": "x"}},
        ], asm2.assemble())

        asm3 = Assembler(
            template_content="`#assistant#`消息内容`*function_call*`{\"name\":\"func1\", \"arguments\":1}"
        )
        self.assertRaises(JiuWenBaseException, asm3.assemble)

        asm4 = Assembler(
            template_content="`#assistant#`消息内容`*function_call*`{\"name\":\"func1\", \"arguments\":\"x\","
                             "\"extra\":\"y\"}"
        )
        self.assertRaises(JiuWenBaseException, asm4.assemble)

        asm5 = Assembler(
            template_content=[
                         {"role": "system", "content": "{{role}}"},
                         {"role": "user", "content": "{{user_inputs}}"},
                         {"role": "assistant", "content": "None", "function_call": {"name": "func1", "arguments": "x"}},
                         {"role": "function", "content": "result of function call", "name": "func1"}
                     ],
            role=TextableVariable(text="你是一个精通{{domain}}领域的问答助手"),
            user_inputs=TextableVariable(text="问题： {{query}}\n答案：")
        )
        self.assertEqual({"domain", "query"}, set(asm5.input_keys))
        self.assertEqual([
            {"role": "system", "content": "你是一个精通科学领域的问答助手"},
            {"role": "user", "content": "问题： 牛顿第三定律\n答案："},
            {"role": "assistant", "content": "None", "function_call": {"name": "func1", "arguments": "x"}},
            {"role": "function", "content": "result of function call", "name": "func1"}
        ], asm5.assemble(domain="科学", query="牛顿第三定律"))

        asm6 = Assembler(
            template_content="""
`#system#`this is a system message
`#assistant#`calling function ... `*function_call*`{"name": "search", "arguments": "{'x': [1,2,3], 'y': '2'}"}
`#function#`this is the result of the function `*name*` search
`#user#`ok"""
        )
        self.assertEqual([
            {"role": "system", "content": "this is a system message"},
            {"role": "assistant", "content": "calling function ...", "function_call":
                {"name": "search", "arguments": "{'x': [1,2,3], 'y': '2'}"}},
            {"role": "function", "content": "this is the result of the function", "name": "search"},
            {"role": "user", "content": "ok"}
        ], asm6.assemble())

    @unittest.skip("李雷修复")
    def test_template_format(self):
        template = Template(
            name="test",
            content="`#system#`你是一个精通{{domain}}领域的问答助手`#user#`{{memory}}")
        template.format({"memory": [{"role": "user", "content": "你是谁"}], "domain": "数学"})
        self.assertEqual(
            template.to_messages(),
            [
                BaseMessage(**{"role": "system", "content": "你是一个精通数学领域的问答助手"}),
                BaseMessage(**{"role": "user", "content": "[{'role': 'user', 'content': '你是谁'}]"})
            ]
        )

        template2 = Template(
                name="xxx",
                content="""
`#system#`this is a system message
`#assistant#`calling function ... `*function_call*`{"name": "search", "arguments": "{'x': [1,2,3], 'y': '2'}"}
`#function#`this is the result of the function `*name*` search
`#user#`ok"""
            )
        template2.format()
        self.assertEqual([
            BaseMessage(**{"role": "system", "content": "this is a system message"}),
            BaseMessage(**{"role": "assistant", "content": "calling function ...", "function_call":
                {"name": "search", "arguments": "{'x': [1,2,3], 'y': '2'}"}}),
            BaseMessage(**{"role": "function", "content": "this is the result of the function", "name": "search"}),
            BaseMessage(**{"role": "user", "content": "ok"})
        ], template2.to_messages())
