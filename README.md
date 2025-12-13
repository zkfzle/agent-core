# openJiuwen Core

## 简介

**openJiuwen Core**是一款开源大语言模型应用开发框架，致力于提供灵活、强大且易用的Agent开发与运行能力。基于该框架，开发者可快速构建处理各类简单或复杂任务的Agent，实现多Agent协同交互，高效开发生产级可靠Agent。助力企业与个人快速搭建Agent系统，推动商用级Agentic AI技术广泛应用与落地。

## 为什么选择openJiuwen Core?

- **开箱即用的组件**：提供丰富的预置组件，包括意图识别、提问器、大模型调用、工具组件等，大幅降低开发门槛。

- **高效精准的任务执行**：内置高性能执行引擎，支持异步并行图执行、组件并发、流式处理等能力，确保Agent在执行任务时的高效性与精准性。

- **灵活可控的多工作流跳转能力**：支持Agent在同一会话中管理多个工作流，支持用户在不同工作流间自由切换，由框架保障被打断工作流的断点接续。解决了用户在同一对话中切换不同任务场景的需求，提供了灵活的多任务管理能力。

- **实用的提示词开发与调优能力**：输入需求即可一键生成适配的提示词，结合真实场景数据集实现自动优化迭代，助力开发者快速产出高质量提示词，降低Agent核心能力开发门槛。

## 快速开始

### 安装

- 操作系统：兼容Windows、Linux、macOS。
- Python 版本：Python的版本应高于或者等于Python 3.11版本，建议使用3.11.4版本，使用前请检查Python版本信息。

**从PyPi安装**

```bash
pip install -U openjiuwen
```

### 样例

让我们创建一个简单的WorkflowAgent，调用工作流生成一段文本:

```python
import os
import asyncio
from openjiuwen.core.component.start_comp import Start
from openjiuwen.core.component.end_comp import End
from openjiuwen.core.component.llm_comp import LLMComponent, LLMCompConfig
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.workflow.workflow_config import WorkflowConfig, WorkflowMetadata, WorkflowInputsSchema
from openjiuwen.core.workflow.base import Workflow
from openjiuwen.core.utils.llm.base import BaseModelInfo
from openjiuwen.agent.workflow_agent.workflow_agent import WorkflowAgent
from openjiuwen.agent.config.workflow_config import WorkflowAgentConfig
from openjiuwen.agent.common.schema import WorkflowSchema

# TODO：请提供用户的大模型配置信息
os.environ.setdefault("API_BASE", "your_api_base")
os.environ.setdefault("API_KEY", "your_api_key")
os.environ.setdefault("MODEL_PROVIDER", "your_provider")
os.environ.setdefault("MODEL_NAME", "your_model_name")
os.environ.setdefault("LLM_SSL_VERIFY", "false")

# 创建大模型配置对象
model_config = ModelConfig(
    model_provider=os.getenv("MODEL_PROVIDER"),
    model_info=BaseModelInfo(
        api_key=os.getenv("API_KEY"),
        api_base=os.getenv("API_BASE"),
        model=os.getenv("MODEL_NAME"),
    )
)

# 创建工作流配置
workflow_config = WorkflowConfig(
    metadata=WorkflowMetadata(
        id="generate_text_workflow",
        name="generate_text",
        version="1.0",
        description="根据用户输入生成文本"
    ),
    workflow_inputs_schema=WorkflowInputsSchema(
        type="object",
        properties={"query": {"type": "string", "description": "用户输入", "required": True}},
        required=['query']
    )
)

# 初始化工作流
flow = Workflow(workflow_config=workflow_config)

# 创建组件
start = Start({"inputs": [{"id": "query", "type": "String", "required": "true", "sourceType": "ref"}]})
end = End({"responseTemplate": "工作流输出文本: {{output}}"})
llm_config = LLMCompConfig(
    model=model_config,
    template_content=[
        {"role": "system", "content": "你是一个AI助手，能够帮我完成任务。\n注意：请不要推理，直接输出结果就好了！"},
        {"role": "user", "content": "{{query}}"}],
    response_format={"type": "text"},
    output_config={"output": {"type": "string", "required": True}},
)
llm = LLMComponent(llm_config)

# 注册组件并连接
flow.set_start_comp("start", start, inputs_schema={"query": "${query}"})
flow.add_workflow_comp("llm", llm, inputs_schema={"query": "${start.query}"})
flow.set_end_comp("end", end, inputs_schema={"output": "${llm.output}"})
flow.add_connection("start", "llm")
flow.add_connection("llm", "end")

# 创建并绑定Agent
schema = WorkflowSchema(
    id=flow.config().metadata.id,
    name=flow.config().metadata.name,
    version=flow.config().metadata.version,
    description="第一个工作流",
    inputs={"query": {"type": "string"}},
)
agent_config = WorkflowAgentConfig(
    id="hello_agent",
    version="0.1.0",
    description="第一个Agent",
    workflows=[schema],
)
workflow_agent = WorkflowAgent(agent_config)
workflow_agent.bind_workflows([flow])

# 运行Agent
async def main():
    invoke_result = await Runner.run_agent(workflow_agent, {"query": "你好,请生成一则笑话,不要超过20个字"})
    output_result = invoke_result.get("output").result
    print(f"WorkflowAgent output result >>> {output_result.get('responseContent')}")

asyncio.run(main())
```

预期输出
```
WorkflowAgent output result >>> 工作流输出文本: 鸡过马路，不会游泳也不会飞。
```


## 架构设计

openJiuwen Core作为openJiuwen架构的核心组成部分，本次开源版本中，其提供的核心能力主要分为三层：

  * **SDK接口层**：为开发者开发Agent提供丰富的SDK接口，具体包含Agent创建，工作流的开发与编排，模型调用与输出解析、提示词构建与填充以及本地与外部工具调用相关接口。
  * **Agent控制与核心组件层**：为开发者提供ReAct与工作流两大场景下的Agent控制能力，具体涵盖复杂任务规划、工具选择与调用、工作流选择与调用以及工作流切换等关键能力。此外，该层还提供意图识别、提问器等开箱即用的各类组件，降低开发门槛。
  * **Agent运行和基础能力层**：为开发者提供高效的Agent执行引擎与强大的运行时环境，同时配套提供Agent运行所需的上下文管理能力及基础工具，保障Agent稳定运行。

## 功能特性

**Agent编排**

openJiuwen致力于提供高效、灵活的Agent应用开发支持，帮助用户快速构建智能化、自动化的Agent应用系统，轻松应对各类复杂任务。目前，openJiuwen提供了ReActAgent和WorkflowAgent这两种预置智能体，提供了丰富的功能和灵活的开发选项，满足用户不同场景下的智能需求。

- ReActAgent：ReActAgent是一种遵循ReAct（Reasoning + Action）规划模式的Agent，通过 "思考（Thought）→ 行动（Action）→ 观察（Observe）"的循环迭代完成用户任务。其强大的多轮推理与自我修正能力，使ReActAgent具备动态决策能力且能够灵活应对环境变化，适用于需要复杂推理和策略调整的多样化任务场景。
- WorkflowAgent：WorkflowAgent是一种专注于多步骤、任务导向的流程自动化Agent，通过严格遵循用户预定义的任务流程高效地执行复杂任务。其侧重于基于预设流程实现任务的规范化与高效化执行，适用于任务结构清晰、可分解为多个步骤的场景。


**高性能执行引擎**

提供高性能执行引擎能力，支持分布式部署和低成本运行，解决海量Agent执行效率低、成本高的问题，有效支撑海量Agent运行和行业生产应用落地。

- 异步并行图执行器：提供组件并发执行、异步IO处理、结构化上下文管理等能力，支持多工作流任务的高效并行处理与异构组件的灵活调用。
- 组件基础能力：提供组件间传值（流批一体）、组件动态跳转、组件状态中断与恢复等基础能力，支持组件动态配置和多实例管理。
- 数据存储和流式处理：提供流式输出、组件间流式等数据控制能力，支持对接外部存储系统实现Agent上下文数据外置，便于分布式场景下的弹性扩展。

## 参与贡献

我们欢迎所有形式的贡献，包括但不限于:
- 提交问题和功能建议
- 改进文档
- 提交代码
- 分享使用经验

## 开源许可证

本项目依据Apache-2.0许可证授权。
