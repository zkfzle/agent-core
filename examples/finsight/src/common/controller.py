import asyncio
import json
import random
import time
import traceback
from typing import List, Callable, Any
from src.common.event import PriorityEvent
from openjiuwen.core.common.logging import logger


class Controller:
    def __init__(self):
        self.tasks: List[PriorityEvent] = []
        self.past_tasks: List[PriorityEvent] = []

    def sort(self):
        self.tasks.sort(key=lambda t: t.priority)

    def add_tasks(self, tasks: List[PriorityEvent], lambda_filter: Callable = None):
        if lambda_filter:
            for task in tasks:
                filter_results = [lambda_filter(task, _task) for _task in self.tasks]
                if all(filter_results):
                    self.tasks.append(task)
        else:
            self.tasks.extend(tasks)

    def get_next_parallel_tasks(self):
        self.refresh()
        if self.tasks:
            current_priority = self.tasks[0].priority
            return [task for task in self.tasks if task.priority == current_priority]
        return None

    async def run_parallel_tasks(self):
        self.refresh()
        if self.tasks:
            try:
                current_priority = self.tasks[0].priority
                tasks = [task for task in self.tasks if task.priority == current_priority and task.inputs]
                running_tasks = []
                for task in tasks:
                    task = asyncio.create_task(task.callable(task))
                    # await task
                    running_tasks.append(task)
                await asyncio.gather(*running_tasks)
            except Exception as e:
                logger.error(f"Error in run_parallel_tasks: {type(e).__name__}: {str(e)}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                raise

    def state(self):
        return {
            "tasks": [t.state() for t in self.tasks],
            "past_tasks": [t.state() for t in self.past_tasks],
        }

    def load_from_state(self, state):
        self.tasks = [PriorityEvent(**t) for t in state["tasks"]]
        self.past_tasks = [PriorityEvent(**t) for t in state["past_tasks"]]

    def refresh(self):
        i = 0
        while i < len(self.tasks):
            if self.tasks[i].status == "finished":
                # 添加到 past_task
                self.past_tasks.append(self.tasks[i])
                # 从 task 中删除
                self.tasks.pop(i)
            else:
                i += 1
        self.sort()
