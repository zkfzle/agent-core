from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.runner.drunner.dmessage_queue.message import DmqResponseMessage, ResultType
from openjiuwen.core.common.logging import logger

# Max queue size per collector
MAX_QUEUE_SIZE = 10000

import asyncio
from typing import Any, AsyncGenerator, Optional

# 取消信号, 用于唤醒等待
CANCEL_MSG = object()


class ResponseCollector:
    """负责收集指定 request 的响应，并支持取消和超时"""

    def __init__(self, message_id: str, receiver_id: str, request_id: str = None, ttl: float = None):
        self.message_id = message_id
        self.receiver_id = receiver_id
        self.request_id = request_id
        self.ttl = ttl or 30.0
        self.queue: asyncio.Queue[DmqResponseMessage | object] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)

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
                self._wake_waiters()
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
            await self.close()
            return

        await self.queue.put(msg)

    async def result(self, timeout: Optional[float] = None) -> Any:
        timeout = timeout or self.ttl

        if self._cancelled:
            raise asyncio.CancelledError(f"Collector({self.message_id}) was cancelled")
        if self._expired:
            raise TimeoutError(f"Collector({self.message_id}) expired")

        try:
            msg = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            # 如果是取消信号
            if msg is CANCEL_MSG:
                raise asyncio.CancelledError(f"Collector({self.message_id}) was cancelled")

            if msg.result_type == ResultType.ERROR:
                raise JiuWenBaseException(msg.error_code, msg.error_msg)
            return msg.payload
        except asyncio.TimeoutError:
            self._expired = True
            await self._cleanup_queue()
            logger.warning(f"[Collector:{self.message_id}] result timeout ({timeout:.1f}s)")
            raise TimeoutError(f"Collector({self.message_id}) timeout waiting for result")
        finally:
            # Close after reading
            await self.close()

    async def stream(self, timeout: Optional[float] = None):
        """流式获取结果"""
        timeout = timeout or self.ttl
        try:
            while True:
                msg = await asyncio.wait_for(self.queue.get(), timeout=timeout)

                if msg is CANCEL_MSG:
                    raise asyncio.CancelledError(f"Collector({self.message_id}) was cancelled")

                if msg.result_type == ResultType.ERROR:
                    raise JiuWenBaseException(msg.error_code, msg.error_msg)

                yield msg.payload

                if msg.last_chunk:
                    break
        except asyncio.TimeoutError:
            self._expired = True
            logger.warning(f"[Collector:{self.message_id}] stream timeout ({timeout:.1f}s)")
            raise TimeoutError(f"Collector({self.message_id}) stream timeout")
        finally:
            await self.close()

    async def close(self):
        """主动取消（包括队列满、系统关闭）"""
        if self._cancelled:
            return

        self._cancelled = True
        if self._expire_task and not self._expire_task.done():
            self._expire_task.cancel()

        await self._cleanup_queue()
        self._wake_waiters()
        logger.info(f"[Collector:{self.message_id}] cancelled")

    async def _cleanup_queue(self):
        """清空队列"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break

    def _wake_waiters(self):
        """往队列放入取消信号以唤醒result/stream接口"""
        try:
            self.queue.put_nowait(CANCEL_MSG)
        except asyncio.QueueFull:
            pass