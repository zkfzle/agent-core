import pytest
from typing import List

from openjiuwen.core.session.stream import OutputSchema
from openjiuwen.core.single_agent import AgentCard
from openjiuwen.core.session import TaskSession
from tests.system_tests.agent.controller_agent.deepsearch import build_deepsearch_agent, TextDataFrame
from openjiuwen.core.controller.schema import ControllerOutputChunk


@pytest.mark.asyncio
async def test_deepsearch_end_to_end_flow():
    """
    测试用例：验证 deepsearch Agent 的端到端完整流程。

    测试目标:
    1.  **触发 handle_input**: 模拟用户输入，启动整个流程。
    2.  **验证 DataCollectTaskExecutor**: 检查数据收集任务是否被执行。
    3.  **验证 handle_task_completion (阶段1->2)**: 确认数据收集完成后，数据分析任务被触发。
    4.  **验证 DataAnalysisTaskExecutor**: 检查数据分析任务是否被执行。
    5.  **验证 handle_task_completion (阶段2->3)**: 确认数据分析完成后，报告生成任务被触发。
    6.  **验证 ReportGenerateTaskExecutor**: 检查报告生成任务是否被执行。
    """
    # 1. Arrange: 构建 Agent 并准备输入
    agent_card = AgentCard(
        id="deepsearch",
        name="DeepSearch",
        description="Arxiv研究报告智能体，可以通过收集、分析数据生成Arxiv研究报告",
    )
    agent = await build_deepsearch_agent(agent_card)
    user_input = "帮我研究一下'芯片'领域的最新进展"
    
    # 创建 session
    session = TaskSession(trace_id="test_deepsearch")

    # 用于收集所有流式输出的文本内容
    output_texts: List[str] = []

    # 2. Act: 执行 Agent 的 stream 方法，并收集所有输出
    # agent.stream() 返回一个异步生成器，我们遍历它来获取所有结果
    async for chunk in agent.stream(user_input, session):
        # chunk 是 ControllerOutputChunk 对象
        # 我们只关心 payload 中的文本数据
        if isinstance(chunk, ControllerOutputChunk):
            if chunk.payload and chunk.payload.data:
                for item in chunk.payload.data:
                    if isinstance(item, TextDataFrame):
                        output_texts.append(item.text)
        else:
            # OutputSchema
            output_texts.append(chunk.payload.get("result", ""))

    # 将所有文本片段合并成一个大字符串，便于搜索验证
    full_output = "\n".join(output_texts)
    print(f"Agent Full Output:\n{full_output}")

    # 3. Assert: 验证流程中的关键输出是否存在
    # 这些断言隐式地验证了 handle_input 和 handle_task_completion 的调用，
    # 因为只有它们被正确调用，后续阶段的任务才会执行并产生预期的输出。

    # 验证数据收集 execute_ability 被调用
    assert "正在收集芯片相关的Arxiv论文数据..." in full_output, "断言失败：数据收集阶段未启动"
    assert "芯片相关Arxiv论文数据收集完成" in full_output, "断言失败：数据收集阶段未报告完成"

    # 验证数据分析 execute_ability 被调用 (由上一个任务的完成事件触发)
    assert "正在分析芯片相关的Arxiv论文数据..." in full_output, "断言失败：数据分析阶段未启动"
    assert "芯片相关Arxiv论文数据分析完成" in full_output, "断言失败：数据分析阶段未报告完成"

    # 验证报告生成 execute_ability 被调用 (由上一个任务的完成事件触发)
    assert "正在生成芯片研究报告..." in full_output, "断言失败：报告生成阶段未启动"
    assert "芯片研究报告生成完成" in full_output, "断言失败：报告生成阶段未报告完成"

    # 验证调用到event_handler的回调函数
    assert "成功调用hanle_input回调" in full_output, "断言失败：没有调用到注册的handle_input函数"
    assert full_output.count("成功调用handle_task_completion回调") == 3, \
        "断言失败：handle_task_completion 回调未被调用 3 次"
