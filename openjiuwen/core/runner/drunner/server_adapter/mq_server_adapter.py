#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Any, Callable, AsyncIterator, Optional

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.logging import logger

from openjiuwen.core.runner.drunner.server_adapter.mq_message_utils import build_stream_response, build_final_response, \
    build_batch_response, build_error_response
from openjiuwen.core.runner.message_queue_base import MessageQueueBase, SubscriptionBase
from openjiuwen.core.runner.drunner.dmessage_queue.message import DmqRequestMessage, DMessageType


@dataclass
class MessageTask:
    message: DmqRequestMessage
    task: asyncio.Task


class MqServerAdapter:
    """负责处理 MQ 请求的 Server Adapter"""

    def __init__(
            self,
            adapter_id: str,
            topic: str,
            invoke_handler: Callable[[dict], Any],
            stream_handler: Callable[[dict], AsyncIterator[Any]],
    ):
        from openjiuwen.core.runner.runner import Runner
        self.adapter_id = adapter_id
        self.topic = topic
        self.invoke_handler = invoke_handler
        self.stream_handler = stream_handler
        self.mq: MessageQueueBase = Runner.distribute_message_queue()
        self.subscription: Optional[SubscriptionBase] = None
        self.active = False
        self._running_tasks: Dict[str, MessageTask] = {}
        self._loop = asyncio.get_event_loop()

    def start(self):
        if not self.active:
            self.subscription = self.mq.subscribe(self.topic)
            self.subscription.set_message_handler(self._handle_message)
            self.subscription.activate()
            self.active = True
            logger.info(f"[{self.adapter_id}] Adapter started on {self.topic}")

    async def _handle_message(self, message: DmqRequestMessage) -> None:
        msg_id = message.message_id

        # 已过期的消息直接丢弃
        if message.expire_at and message.expire_at < time.time():
            logger.warning(f"[{self.adapter_id}] Ignoring expired message {msg_id}")
            return

        # 被动取消
        if message.type == DMessageType.STOP:
            await self._cancel_task(msg_id, inner_cancel=False)
            return

        # 重复消息
        if msg_id in self._running_tasks:
            logger.warning(f"[{self.adapter_id}] Duplicate msg_id {msg_id}, replacing old task")
            await self._cancel_task(msg_id, inner_cancel=True)

        # 启动执行任务
        task = asyncio.create_task(self._process_message(message))
        self._running_tasks[msg_id] = MessageTask(message, task)
        task.add_done_callback(lambda t: self._cleanup_task(msg_id, t))

        # 定时超时取消
        if message.expire_at:
            when = message.expire_at
            delay = when - time.time()
            if delay > 0:
                self._loop.call_at(
                    self._loop.time() + delay,
                    lambda: self._timeout_cancel(msg_id)
                )

    async def _process_message(self, message: DmqRequestMessage):
        try:
            if message.enable_stream:
                seq = 0
                async for chunk in self.stream_handler(message.payload):
                    resp = build_stream_response(message, self.adapter_id, chunk, seq, last=False)
                    await self.mq.produce_message(message.reply_topic, resp)
                    seq += 1

                final_resp = build_final_response(message, self.adapter_id, seq)
                await self.mq.produce_message(message.reply_topic, final_resp)
            else:
                result = await self.invoke_handler(message.payload)
                resp = build_batch_response(message, self.adapter_id, result)
                await self.mq.produce_message(message.reply_topic, resp)


        except asyncio.CancelledError:
            # 取消异常只有server内部因为adapter stop取消需要发error code给客户端
            logger.info(f"[{self.adapter_id}] Task {message.message_id} cancelled")
            raise

        except JiuWenBaseException as e:
            # runner、mq都先于adapter关闭
            if (e.error_code == StatusCode.RUNNER_STOPPED.code
                    or e.error_code == StatusCode.MESSAGE_QUEUE_NOT_RUNNING.code):
                logger.info(f"[{self.adapter_id}] Task {message.message_id} cancelled")
                raise
            # Runner.run的执行异常，需要返回给客户端
            logger.warning(f"[{self.adapter_id}] adapter run error msg: {message.message_id}: {e}")
            resp = build_error_response(message, self.adapter_id, e)
            await self.mq.produce_message(message.reply_topic, resp)

        except Exception as e:
            # Runner.run返回了不可预期的异常，需要返回给客户端
            logger.exception(f"[{self.adapter_id}] Unexpected error: {e}")
            err = JiuWenBaseException(StatusCode.RUNNER_STOPPED.code, str(e))
            resp = build_error_response(message, self.adapter_id, err)
            await self.mq.produce_message(message.reply_topic, resp)

    def _timeout_cancel(self, msg_id: str):
        msg_task = self._running_tasks.get(msg_id)
        if not msg_task:
            return
        task = msg_task.task
        if not task.done():
            logger.warning(f"[{self.adapter_id}] Task {msg_id} expired and will be cancelled")
            task.cancel()

    async def _cancel_task(self, msg_id: str, *, inner_cancel: bool):
        msg_task = self._running_tasks.get(msg_id)
        if not msg_task:
            return
        message, task = msg_task.message, msg_task.task

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._running_tasks.pop(msg_id, None)
            if inner_cancel:
                try:
                    err = JiuWenBaseException(
                        StatusCode.RUNNER_STOPPED.code,
                        f"Task cancelled by adapter stop ({self.adapter_id})"
                    )
                    resp = build_error_response(message, self.adapter_id, err)
                    await self.mq.produce_message(message.reply_topic, resp)
                except Exception as e:
                    logger.warning(f"[{self.adapter_id}] Failed to send cancel error: {e}")

    def _cleanup_task(self, msg_id: str, task: asyncio.Task):
        """任务结束清理"""
        self._running_tasks.pop(msg_id, None)
        if task.cancelled():
            logger.info(f"[{self.adapter_id}] Task {msg_id} cancelled (cleanup)")
        elif exc := task.exception():
            logger.error(f"[{self.adapter_id}] Task {msg_id} failed: {exc}")

    async def stop(self):
        """停止adapter"""
        logger.info(f"[{self.adapter_id}] Stopping adapter...")
        if not self.active:
            return
        self.active = False

        if self.subscription:
            await self.subscription.deactivate()
            self.subscription = None

        logger.info(f"[{self.adapter_id}] Cancelling all running tasks...")
        await asyncio.gather(
            *[self._cancel_task(mid, inner_cancel=True) for mid in list(self._running_tasks.keys())],
            return_exceptions=True,
        )
        logger.info(f"[{self.adapter_id}] Adapter stopped")

    def get_running_tasks_count(self) -> int:
        return len(self._running_tasks)
