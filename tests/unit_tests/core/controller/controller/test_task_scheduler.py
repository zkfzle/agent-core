"""TaskScheduler 功能测试脚本

测试 TaskScheduler 的核心功能是否正常工作。
"""
import asyncio, pytest
from openjiuwen.core.controller.modules.task_scheduler import TaskScheduler, TaskExecutor
from openjiuwen.core.controller.modules.task_manager import TaskManager
from openjiuwen.core.controller.modules.event_queue import EventQueue
from openjiuwen.core.controller.config import ControllerConfig
from openjiuwen.core.controller.schema.task import Task, TaskStatus
from openjiuwen.core.controller.schema.controller_output import ControllerOutputChunk, ControllerOutputPayload
from openjiuwen.core.controller.schema.dataframe import TextDataFrame
from openjiuwen.core.controller.schema.event import EventType
from openjiuwen.core.context_engine import ContextEngine
from openjiuwen.core.session import TaskSession
from openjiuwen.core.single_agent.agent import AbilityKit
from typing import AsyncIterator


class DummyTaskExecutor(TaskExecutor):
    """测试用的简单任务执行器"""
    
    async def execute_ability(self, task_id: str, session) -> AsyncIterator[ControllerOutputChunk]:
        """执行任务"""
        # 模拟处理中
        yield ControllerOutputChunk(
            index=0,
            type="controller_output",
            payload=ControllerOutputPayload(
                type="processing",
                data=[TextDataFrame(type="text", text=f"Task {task_id} is processing...")]
            ),
            last_chunk=False
        )
        
        await asyncio.sleep(0.5)  # 模拟耗时操作
        
        # 任务完成
        yield ControllerOutputChunk(
            index=1,
            type="controller_output",
            payload=ControllerOutputPayload(
                type=EventType.TASK_COMPLETION,
                data=[TextDataFrame(type="text", text=f"Task {task_id} completed")]
            ),
            last_chunk=True
        )
    
    async def can_pause(self, task_id: str, session) -> tuple[bool, str]:
        return True, ""
    
    async def pause(self, task_id: str, session) -> bool:
        return True
    
    async def can_cancel(self, task_id: str, session) -> tuple[bool, str]:
        return True, ""
    
    async def cancel(self, task_id: str, session) -> bool:
        return True


def build_dummy_executor(config, ability_kit, context_engine, task_manager, event_queue):
    """构建测试执行器"""
    return DummyTaskExecutor(config, ability_kit, context_engine, task_manager, event_queue)

@pytest.mark.asyncio
async def test_task_scheduler():
    """测试 TaskScheduler 基本功能"""
    print("=" * 60)
    print("测试 TaskScheduler 基本功能")
    print("=" * 60)
    
    # 1. 创建依赖
    config = ControllerConfig()
    task_manager = TaskManager(config)
    event_queue = EventQueue(config)
    context_engine = ContextEngine()
    ability_kit = AbilityKit()
    
    # 2. 创建 TaskScheduler
    scheduler = TaskScheduler(
        config=config,
        task_manager=task_manager,
        context_engine=context_engine,
        ability_kit=ability_kit,
        event_queue=event_queue,
        agent_id="test_agent"
    )
    
    # 3. 注册任务执行器
    scheduler.task_executor_registry.add_task_executor("test_task", build_dummy_executor)
    
    # 4. 创建会话
    session = TaskSession(trace_id="test_session")
    session_id = session.session_id()
    scheduler.sessions[session_id] = session
    
    # 5. 创建测试任务
    task1 = Task(
        session_id=session_id,
        task_id="task_1",
        task_type="test_task",
        description="Test task 1",
        priority=1,
        status=TaskStatus.SUBMITTED
    )
    
    task2 = Task(
        session_id=session_id,
        task_id="task_2",
        task_type="test_task",
        description="Test task 2",
        priority=1,
        status=TaskStatus.SUBMITTED
    )
    
    task_manager.add_task([task1, task2])
    print(f"✅ 创建了 2 个测试任务")
    
    # 6. 启动调度器
    await scheduler.start()
    print(f"✅ TaskScheduler 已启动")
    
    # 7. 等待任务完成
    await asyncio.sleep(2)
    
    # 8. 检查任务状态
    task1_status = task_manager.get_task("task_1").status
    task2_status = task_manager.get_task("task_2").status
    
    print(f"\n任务状态:")
    print(f"  Task 1: {task1_status}")
    print(f"  Task 2: {task2_status}")
    
    # 9. 停止调度器
    await scheduler.stop()
    print(f"\n✅ TaskScheduler 已停止")
    
    # 10. 验证结果
    if task1_status == TaskStatus.COMPLETED and task2_status == TaskStatus.COMPLETED:
        print("\n" + "=" * 60)
        print("✅ 测试通过！所有任务都成功完成")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("❌ 测试失败！任务未正确完成")
        print("=" * 60)
        return False


async def main():
    """主函数"""
    try:
        success = await test_task_scheduler()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    asyncio.run(test_task_scheduler())
