import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from openjiuwen.core.runtime.runtime import BaseRuntime
from openjiuwen.graph.checkpoint.base import create_checkpoint, PendingNode
from openjiuwen.graph.checkpoint.gragh_checkpoiter import GraphCheckpointer
from openjiuwen.graph.checkpoint.memory_checkpoint_saver import MemoryCheckpointSaver


@dataclass
class Message:
    value: Any

class DummyChannel:
    def __init__(self, value):
        self.value = value

    def checkpoint(self):
        return self.value

    def snapshot(self):
        return self.value

async def _test_memory_checkpoint_saver():

    saver = MemoryCheckpointSaver()

    conversation_id = "conv_123"
    ns = "default"

    # Create a mock checkpoint
    checkpoint = create_checkpoint(
        ns=ns,
        step=1,
        channel_snapshot={"ch1": 42},
        pending_buffer=[Message("pending msg")],
        pending_node={
            "node1": PendingNode(node_name="n1", status="running")
        }
    )

    # ---- Save checkpoint ----

    await saver.save(conversation_id, ns, checkpoint)
    print("[TEST] save() executed.")

    # ---- Get checkpoint ----
    loaded = await saver.get(conversation_id, ns)
    assert loaded is not None, "Loaded checkpoint should not be None."
    assert loaded.step == 1
    assert loaded.channel_values["ch1"] == 42
    assert loaded.pending_buffer[0].value == "pending msg"
    assert loaded.pending_node["node1"].status == "running"
    assert loaded.pending_node["node1"].node_name == "n1"
    print("[TEST] get() verified values OK.")

    # ---- Delete conversation ----
    await saver.delete(conversation_id)
    deleted = await saver.get(conversation_id, ns)
    assert deleted is None, "After delete, checkpoint should be None."

    print("[TEST] delete() OK.")
    print("\nAll MemoryCheckpointSaver tests passed!")

async def _test_memory_graph_checkpointer():
    saver = MemoryCheckpointSaver()
    mock_runtime = MagicMock(spec=BaseRuntime)
    graph_checkpoint = GraphCheckpointer(runtime=mock_runtime, saver=saver)

    conversation_id = "conv_321"
    ns = "default_ns"

    # Create a mock checkpoint
    checkpoint = create_checkpoint(
        ns=ns,
        step=2,
        channel_snapshot={"ch1": 25},
        pending_buffer=[Message("pending msg2")],
        pending_node={
            "node1": PendingNode(node_name="n2", status="running2")
        }
    )

    await graph_checkpoint.save(conversation_id, ns, checkpoint)



    loaded = await saver.get(conversation_id, ns)
    assert loaded is not None, "Loaded checkpoint should not be None."
    assert loaded.step == 2
    assert loaded.channel_values["ch1"] == 25
    assert loaded.pending_buffer[0].value == "pending msg2"
    assert loaded.pending_node["node1"].status == "running2"
    assert loaded.pending_node["node1"].node_name == "n2"

    # ---- Delete conversation ----
    await saver.delete(conversation_id, ns)
    deleted = await saver.get(conversation_id, ns)
    assert deleted is None, "After delete, checkpoint should be None."
    print("[TEST] delete() OK.")
    print("\nAll test_memory_graph_checkpointer tests passed!")

def test_memory_checkpoint_saver_basic():
    asyncio.run(_test_memory_checkpoint_saver())
    asyncio.run(_test_memory_graph_checkpointer())


if __name__ == "__main__":
    test_memory_checkpoint_saver_basic()
