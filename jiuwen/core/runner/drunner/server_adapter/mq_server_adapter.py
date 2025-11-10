import asyncio
from typing import Callable, Any, AsyncIterator, Optional, Dict

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.common.logging import logger
from jiuwen.core.runner.drunner.dmessage_queue.dsubscription.subscription import DSubscription
from jiuwen.core.runner.drunner.dmessage_queue.message import DmqRequestMessage, DMessageType, DmqResponseMessage, \
    ResultType
from jiuwen.core.runner.runner import Runner


class MqServerAdapter:
    """MqServerAdapter core"""

    def __init__(
            self,
            adapter_id: str,
            topic: str,
            invoke_handler: Callable[[dict], Any],
            stream_handler: Callable[[dict], AsyncIterator[Any]]
    ):
        self.adapter_id = adapter_id
        self.topic = topic
        self.invoke_handler = invoke_handler
        self.stream_handler = stream_handler
        self.mq = Runner._mq
        self.subscription: Optional[DSubscription] = None
        self.active = False
        self._running_tasks: Dict[str, asyncio.Task] = {}  # message_id -> task

    async def _handle_request(self, message: DmqRequestMessage) -> None:
        """处理接收到的MQ消息（异步，不阻塞）"""
        message_id = message.message_id

        # 检查是否是停止消息
        if message.type == DMessageType.STOP:
            await self._cancel_task(message_id)
            return

        # 创建异步任务处理请求(不等待完成，立即返回)
        task = asyncio.create_task(self._process_request(message))
        self._running_tasks[message_id] = task

        # 任务完成后自动清理
        task.add_done_callback(lambda t: self._cleanup_task(message_id, t))

    async def _process_request(self, message: DmqRequestMessage):
        """实际处理请求逻辑"""
        try:
            # 从payload中提取参数
            enable_stream = message.enable_stream
            inputs = message.payload

            if enable_stream:
                # 流式处理
                result_stream = self.stream_handler(inputs)
                await self._send_stream_response(message, result_stream)
            else:
                # 批式处理
                result = await self.invoke_handler(inputs)
                await self._send_batch_response(message, result)

        except asyncio.CancelledError:
            logger.info(f"[{self.adapter_id}] Task {message.message_id} cancelled")
            raise
        except JiuWenBaseException as e:
            logger.error(f"[{self.adapter_id}] Error processing {message.message_id}: {e}")
            await self._send_error_response(message, e)

    async def _send_stream_response(self, message: DmqRequestMessage, result_stream: AsyncIterator):
        """发送流式响应"""
        seq = 0
        try:
            async for chunk in result_stream:
                resp = DmqResponseMessage(
                    type=DMessageType.OUTPUT,
                    message_id=message.message_id,
                    payload=chunk,
                    sender_id=self.adapter_id,
                    receiver_id=message.sender_id,
                    seq=seq,
                    last_chunk=False
                )
                await self.mq.produce_message(message.reply_topic, resp)
                seq += 1

            # 发送最后一条消息
            final_resp = DmqResponseMessage(
                type=DMessageType.OUTPUT,
                message_id=message.message_id,
                payload="",
                sender_id=self.adapter_id,
                receiver_id=message.sender_id,
                seq=seq,
                last_chunk=True
            )
            await self.mq.produce_message(message.reply_topic, final_resp)

        except JiuWenBaseException as e:
            logger.error(f"[{self.adapter_id}] Stream error: {e}")
            await self._send_error_response(message, e)

    async def _send_batch_response(self, message: DmqRequestMessage, result: Any):
        """发送批式响应"""
        resp = DmqResponseMessage(
            type=DMessageType.OUTPUT,
            message_id=message.message_id,
            payload=result,
            sender_id=self.adapter_id,
            receiver_id=message.sender_id,
            seq=0,
            last_chunk=True
        )
        await self.mq.produce_message(message.reply_topic, resp)

    async def _send_error_response(self, message: DmqRequestMessage, error: JiuWenBaseException):
        """发送错误响应"""
        resp = DmqResponseMessage(
            type=DMessageType.OUTPUT,
            message_id=message.message_id,
            payload={},
            result_type=ResultType.ERROR,
            error_code=error.error_code,
            error_msg=error.message,
            sender_id=self.adapter_id,
            receiver_id=message.sender_id,
            seq=0,
            last_chunk=True
        )
        await self.mq.produce_message(message.reply_topic, resp)

    async def _cancel_task(self, message_id: str):
        """取消指定任务"""
        if message_id in self._running_tasks:
            task = self._running_tasks[message_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                if message_id in self._running_tasks:
                    del self._running_tasks[message_id]

    def _cleanup_task(self, message_id: str, task: asyncio.Task):
        """清理任务"""
        if message_id in self._running_tasks:
            del self._running_tasks[message_id]

        if task.exception() and not task.cancelled():
            logger.error(f"[{self.adapter_id}] Task {message_id} failed: {task.exception()}")

    def start(self):
        """启动服务"""
        if not self.active:
            self.subscription = self.mq.subscribe(self.topic)
            self.subscription.set_message_handler(self._handle_request)
            self.subscription.activate()
            self.active = True
            logger.info(f"[{self.adapter_id}] Adapter started, listening on: {self.topic}")

    def stop(self):
        """停止服务"""
        if self.active:
            # 取消所有运行中的任务
            for message_id in list(self._running_tasks.keys()):
                asyncio.create_task(self._cancel_task(message_id))

            if self.subscription:
                self.subscription.deactivate()
                self.subscription = None
            self.active = False
            logger.info(f"[{self.adapter_id}] Adapter stopped")
