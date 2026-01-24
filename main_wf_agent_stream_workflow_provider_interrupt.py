# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import asyncio
import os

from openjiuwen.core.application.workflow_agent import WorkflowAgent
from openjiuwen.core.foundation.llm import ModelRequestConfig, ModelClientConfig
from openjiuwen.core.runner import Runner
from openjiuwen.core.session import InteractiveInput
from openjiuwen.core.session.stream import OutputSchema
from openjiuwen.core.single_agent.legacy import WorkflowAgentConfig, workflow_provider
from openjiuwen.core.workflow import WorkflowCard, generate_workflow_key, QuestionerComponent, QuestionerConfig, \
    FieldInfo, Start, End, Workflow

API_BASE = os.getenv("API_BASE", "https://api.siliconflow.cn/v1/chat/completions")
API_KEY = os.getenv("API_KEY", "sk-kbtegacviizglhrjuyysgvrqlhdrkhbrtmjoqfxvhneidhfu")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "SiliconFlow")
os.environ.setdefault("LLM_SSL_VERIFY", "false")
os.environ.setdefault("IS_SENSITIVE", "false")

class Utils:
    @staticmethod
    def create_model_client_config():
        model_client = ModelClientConfig(
            client_provider=MODEL_PROVIDER,
            api_base=API_BASE,
            api_key=API_KEY,
            verify_ssl=False
        )
        return model_client

    @staticmethod
    def create_model_config():
        return ModelRequestConfig(
            model=MODEL_NAME
        )

    @staticmethod
    def build_workflow_with_questioner():
        workflow_card = WorkflowCard(
            id="questioner_workflow",
            name="questioner_workflow",
            version="1.0",
            input_params={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "用户输入"}},
                "required": ['query']
            }
        )
        flow = Workflow(card=workflow_card)

        start_component = Start()
        end_component = End({"responseTemplate": "姓名：{{name}} | 年龄：{{age}}"})

        key_fields = [
            FieldInfo(field_name="name", description="用户姓名", required=True),
            FieldInfo(field_name="age", description="用户年龄", required=True)
        ]
        questioner_config = QuestionerConfig(
            model_client_config=Utils.create_model_client_config(),
            model_config=Utils.create_model_config(),
            question_content="",
            extract_fields_from_response=True,
            field_names=key_fields,
        )
        questioner_component = QuestionerComponent(questioner_comp_config=questioner_config)

        flow.set_start_comp("s", start_component, inputs_schema={"query": "${query}"})
        flow.set_end_comp("e", end_component,
                          inputs_schema={"name": "${questioner.name}", "age": "${questioner.age}"})
        flow.add_workflow_comp("questioner", questioner_component, inputs_schema={"query": "${s.query}"})

        flow.add_connection("s", "questioner")
        flow.add_connection("questioner", "e")

        return flow

    @staticmethod
    def create_workflow_agent(workflow):
        agent_config = WorkflowAgentConfig(
            id="提问器_agent",
            version="0.1.1",
            description="用户信息收集agent",
            workflows=[]
        )
        agent = WorkflowAgent(agent_config)

        inputs_schem_dict = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用户输入信息"
                }
            },
            "required": ['query']
        }

        @workflow_provider(workflow_id="questioner_workflow", workflow_name="questioner_workflow",
                           workflow_version="0.0.1",
                           workflow_description="用户信息收集agent",
                           inputs=inputs_schem_dict)
        def create_interrupt_workflow_instance():
            """工厂函数：每次调用创建新的 workflow 实例"""
            result = Utils.build_workflow_with_questioner()
            return result

        agent.add_workflows([create_interrupt_workflow_instance])

        return agent


async def main():
    my_workflow = Utils.build_workflow_with_questioner()
    Runner.resource_mgr.add_workflow(
        WorkflowCard(id=generate_workflow_key(my_workflow.card.id, my_workflow.card.version)),
        lambda: my_workflow)
    workflow_agent = Utils.create_workflow_agent(my_workflow)
    interaction_output_schema = []
    async for chunk in Runner.run_agent_streaming(agent=workflow_agent,
            inputs={"conversation_id": "12345", "query": "请帮我收集和整理用户信息。这名用户的姓名是张三"}):
        print(f"WorkflowAgent 第一次输出结果 >>> {chunk}")
        if isinstance(chunk, OutputSchema) and chunk.type == "__interaction__":
            interaction_output_schema.append(chunk)

    await asyncio.sleep(3)

    if interaction_output_schema:
        user_input = InteractiveInput()
        for item in interaction_output_schema:
            component_id = item.payload.id
            if component_id == "questioner":
                user_input.update(component_id, "用户的年龄是25岁")
        async for chunk in Runner.run_agent_streaming(workflow_agent, {"conversation_id": "12345", "query": user_input}):
            print(f"WorkflowAgent 第二次输出结果 >>> {chunk}")


if __name__ == "__main__":
    asyncio.run(main())
