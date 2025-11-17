#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.runner.drunner.dmessage_queue.message import DmqResponseMessage, ResultType
from openjiuwen.core.common.logging import logger

# Max queue size per collector
MAX_QUEUE_SIZE = 10000


class CancelReason(str, Enum):
    """取消原因：用于区分唤醒后的异常类型"""
    RUNNER_STOPPED = "runner_stopped"  # Runner/Adapter 主动停止（应抛 RUNNER_STOPPED）
    TTL_EXPIRE = "ttl_expire"  # TTL 到期（应抛 TimeoutError）
    QUEUE_FULL = "queue_full"  # 队列满（应该抛 CancelledError）
    FINISH = "finish"  # 正常接受完不需要唤醒


@dataclass(frozen=True)
class CancelEvent:
    reason: CancelReason
    info: Optional[str] = None


class ResponseCollector:
    """负责收集指定 request 的响应，并支持取消和超时"""

    def __init__(self, message_id: str, receiver_id: str, request_id: str = None, ttl: float = None):
        self.message_id = message_id
        self.receiver_id = receiver_id
        self.request_id = request_id
        self.ttl = ttl or 30.0
        self.queue: asyncio.Queue[DmqResponseMessage | CancelEvent] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)

        self._cancelled = False
        self._expired = False

        # Start TTL expiration task
        self._expire_task = asyncio.create_task(self._expire_after_ttl())

    def is_cancelled(self) -> bool:
        return self._cancelled

    def is_expired(self) -> bool:
        return self._expired

    def is_active(self) -> bool:
        return not (self._cancelled or self._expired)

    async def _expire_after_ttl(self):
        """TTL 到期自动标记过期"""
        try:
            await asyncio.sleep(self.ttl)
            if not self._cancelled:
                self._expired = True
                await self._cleanup_queue()
                logger.warning(f"[Collector:{self.message_id}] expired after {self.ttl:.1f}s")
                # 唤醒阻塞的的等待请求
                self._wake_waiters(CancelEvent(CancelReason.TTL_EXPIRE))
        except asyncio.CancelledError:
            # 被主动关闭，不记录为过期
            return

    async def put_message(self, msg: DmqResponseMessage):
        """从replyTopic接收消息"""
        if not self.is_active():
            logger.warning(f"[Collector:{self.message_id}] inactive, discard message")
            return

        if self.queue.full():
            logger.warning(f"[Collector:{self.message_id}] queue full({MAX_QUEUE_SIZE}), auto-cancelled")
            await self.close(reason=CancelReason.QUEUE_FULL)
            return

        await self.queue.put(msg)

    async def result(self, timeout: Optional[float] = None) -> Any:
        timeout = timeout or self.ttl

        if self._cancelled:
            raise asyncio.CancelledError(f"Collector({self.message_id}) was cancelled before request send")
        if self._expired:
            raise TimeoutError(f"Collector({self.message_id}) expired")
        try:
            msg = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            await self.check_message(msg)

            return msg.payload
        except asyncio.TimeoutError:
            self._expired = True
            await self._cleanup_queue()
            logger.warning(f"[Collector:{self.message_id}] result timeout ({timeout:.1f}s)")
            raise TimeoutError(f"Collector({self.message_id}) timeout waiting for result")
        except Exception as e:
            raise e
        finally:
            await self.close(reason=CancelReason.FINISH)

    async def stream(self, timeout: Optional[float] = None):
        """流式获取结果"""
        timeout = timeout or self.ttl
        try:
            while True:
                msg = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                logger.debug(f"[Collector:{self.message_id}] stream get message {msg}")
                await self.check_message(msg)
                if msg.last_chunk:
                    # 最后一条是mq空标记，不返回
                    break
                yield msg.payload
        except asyncio.TimeoutError:
            self._expired = True
            logger.warning(f"[Collector:{self.message_id}] stream timeout ({timeout:.1f}s)")
            raise TimeoutError(f"Collector({self.message_id}) stream timeout")
        except Exception as e:
            raise e
        finally:
            await self.close(reason=CancelReason.FINISH)

    async def check_message(self, msg: DmqResponseMessage | CancelEvent):
        if isinstance(msg, CancelEvent):
            logging.info(f"[Collector:{self.message_id}] rev CancelEvent stream cancelled by {msg.reason}")
            if msg.reason == CancelReason.TTL_EXPIRE:
                # TTL 到期 → 抛 TimeoutError
                raise TimeoutError(f"Collector({self.message_id}) timeout")
            elif msg.reason == CancelReason.QUEUE_FULL:
                # 返回消息队列满了但是没来取 客户端应该不要了
                raise asyncio.CancelledError(f"Collector({self.message_id}) queue full")
            else:
                raise JiuWenBaseException(StatusCode.RUNNER_STOPPED.code,
                                          StatusCode.RUNNER_STOPPED.errmsg.format(
                                              "Collector({self.message_id}) was cancelled"))
        if msg.result_type == ResultType.ERROR:
            # 远端的错误码封装到错误消息里
            raise JiuWenBaseException(StatusCode.REMOTE_AGENT_PROCESS_ERROR.code,
                                      StatusCode.REMOTE_AGENT_PROCESS_ERROR.errmsg.format(
                                          error_code=msg.error_code, error_msg=msg.error_msg))

    async def close(self, reason: CancelReason = CancelReason.RUNNER_STOPPED):
        """主动取消（包括队列满、系统关闭）"""
        if self._cancelled:
            return

        self._cancelled = True
        if self._expire_task and not self._expire_task.done():
            self._expire_task.cancel()

        await self._cleanup_queue()
        if reason != CancelReason.FINISH:
            self._wake_waiters(CancelEvent(reason))
            logger.info(f"[Collector:{self.message_id}] cancelled by close({reason})")
        logger.info(f"[Collector:{self.message_id}] cancelled by finished")

    async def _cleanup_queue(self):
        """清空队列"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break

    def _wake_waiters(self, cancel_event: CancelEvent):
        """往队列放入取消信号以唤醒result/stream接口"""
        try:
            self.queue.put_nowait(cancel_event)
        except asyncio.QueueFull:
            pass
