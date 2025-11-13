from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, AsyncGenerator, Sequence, Union

from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.multi_agent.stream.base import StreamData, StreamCode, StreamDataMsg


class MemberMessageType(Enum):
    """Member return message type"""
    STREAM = "stream"  # Stream output
    INTERRUPT = "interrupt"  # Interrupt and wait for user
    ERROR = "error"


@dataclass
class Message:
    """Unified message structure"""
    type: MemberMessageType
    agent_id: str
    data: Union[StreamData]
    timestamp: float = None
    metadata: Dict[str, Any] = None


class Member:
    """Base class for members in multi-Agent Runner, supports direct wrapping of Agent instances"""

    def __init__(self, member_id: str, agent_instance: Any = None):
        self._member_id = member_id
        self._run_state: Optional['RunState'] = None
        self._agent = agent_instance

    @property
    def member_id(self) -> str:
        """Get member ID"""
        return self._member_id

    @staticmethod
    def _is_interrupt_message(chunk: Any) -> bool:
        """Determine if it's an interrupt message"""
        if isinstance(chunk, StreamData) and chunk.code == StreamCode.CONTROLLER_AGENT_INTERRUPT_MESSAGE.value:
            return True
        return False

    async def on_message(self, message: Any, sender: Optional[str] = None, message_id: Optional[str] = None) -> Any:
        """
        Process received message

        Args:
            message: Message content
            sender: Sender ID
            message_id: Message ID

        Returns:
            Processing result
        """
        if self._agent is not None:
            try:
                if isinstance(message, dict):
                    inputs = message.get('inputs', message)
                    kwargs = message.get('kwargs', {})
                else:
                    inputs = message
                    kwargs = {}

                # Call Agent's astream method
                result = []
                async for chunk in self._agent.astream(inputs, **kwargs):
                    result.append(chunk)
                return result
            except Exception as e:
                raise JiuWenBaseException(
                    error_code=StatusCode.MULTI_AGENT_MEMBER_PROCESS_MESSAGE_ERROR.code,
                    message=f"Failed to process message in member {self._member_id}: {str(e)}",
                ) from e
        else:
            # If no Agent instance, return default response
            return f"Member {self._member_id} received message: {message}"

    async def on_messages_stream(
            self,
            messages: Sequence[Any],
            context: Dict[str, Any] = None
    ) -> AsyncGenerator[Message, None]:
        """
        Process message sequence and return Message stream

        Args:
            messages: Message sequence
            context: Context information

        Yields:
            Message: Unified message structure
        """
        if not self._agent:
            return

        for message in messages:
            try:
                # Call Agent's astream method and wrap as Message

                async for chunk in self._agent.stream(**message):
                    # Check if it's a special control message
                    if self._is_interrupt_message(chunk):
                        yield Message(
                            type=MemberMessageType.INTERRUPT,
                            agent_id=self._member_id,
                            data=chunk,
                            metadata={"source_message": message}
                        )
                    else:
                        # Normal stream output
                        yield Message(
                            type=MemberMessageType.STREAM,
                            agent_id=self._member_id,
                            data=chunk,  # Assume chunk is StreamData type
                            metadata={"source_message": message}
                        )

            except JiuWenBaseException:
                raise
            except Exception as e:
                error_msg = f"Error occurred while processing message: {str(e)}"
                yield Message(
                    type=MemberMessageType.ERROR,
                    agent_id=self._member_id,
                    data=StreamData(
                        code=StreamCode.ERROR,
                        msg=StreamDataMsg.FAIL.value,
                        data=dict(
                            message=error_msg,
                            node_name="控制器",
                        ),
                        execution_id=""
                    ),
                    metadata={"error": True, "source_message": message, "exception": error_msg}
                )
