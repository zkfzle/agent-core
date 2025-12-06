#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import asyncio

import pytest

from openjiuwen.core.runtime.interaction.checkpointer import default_inmemory_checkpointer
from openjiuwen.graph.pregel.builder import PregelGraphBuilder
from openjiuwen.graph.pregel.channels import TriggerChannel, BarrierChannel
from openjiuwen.graph.pregel.config import PregelConfig
from openjiuwen.graph.pregel.constants import START, END
from openjiuwen.graph.pregel.engine import Pregel
from openjiuwen.graph.pregel.nodes import PregelNode
from openjiuwen.graph.pregel.router import StaticRouter, BarrierRouter, ConditionalRouter


@pytest.fixture
def basic_nodes_and_channels_direct():
    """Fixture providing a basic graph with barrier synchronization using direct construction.

    Graph structure:
    start -> a -> a1 --\
            b --------\
            c ---------\-> collect -> end
            d --------/
    """

    # Node functions
    def fn_pass():
        return "pass"

    def fn_slow():
        return "slow_data"

    a1bcd_to_collect = BarrierChannel("collect", expected={"a1", "b", "c", "d"})

    # Define channels
    channels = [
        TriggerChannel("start"),
        TriggerChannel("a"),
        TriggerChannel("b"),
        TriggerChannel("c"),
        TriggerChannel("d"),
        TriggerChannel("a1"),
        TriggerChannel("end"),
        a1bcd_to_collect
    ]

    # Define nodes and routers
    nodes = {
        "start": PregelNode("start", fn_pass, [StaticRouter(["a", "b", "c", "d"])]),  # 1->4 Fan-out
        "a": PregelNode("a", fn_pass, [StaticRouter(["a1"])]),
        "b": PregelNode("b", fn_pass, [BarrierRouter([a1bcd_to_collect.key])]),  # Writes to barrier
        "c": PregelNode("c", fn_pass, [BarrierRouter([a1bcd_to_collect.key])]),
        "d": PregelNode("d", fn_pass, [BarrierRouter([a1bcd_to_collect.key])]),
        "a1": PregelNode("a1", fn_slow, [BarrierRouter([a1bcd_to_collect.key])]),  # The delayed input to barrier
        "collect": PregelNode("collect", fn_pass, [StaticRouter(["end"])]),
        "end": PregelNode("end", fn_pass, [StaticRouter([])]),
    }

    return nodes, channels


@pytest.fixture
def basic_nodes_and_channels_builder():
    """Fixture providing a basic graph with barrier synchronization using builder.

    Graph structure:
    start -> a -> a1 --\
            b --------\
            c ---------\-> collect -> end
            d --------/
    """

    def fn_pass():
        print("pass!!!")
        return "pass"

    def fn_slow(): return "slow_data"

    builder = PregelGraphBuilder()
    builder.add_node("start", fn_pass)
    builder.add_node("a", fn_pass)
    builder.add_node("b", fn_pass)
    builder.add_node("c", fn_pass)
    builder.add_node("d", fn_pass)
    builder.add_node("a1", fn_slow)
    builder.add_node("collect", fn_pass)
    builder.add_node("end", fn_pass)

    builder.add_edge("start", ["a", "b", "c", "d"])
    builder.add_edge("a", "a1")
    builder.add_edge(["a1", "b", "c", "d"], "collect")
    builder.add_edge("collect", "end")

    return builder.nodes, builder.channels


@pytest.fixture(params=[
    ("direct", "basic_nodes_and_channels_direct"),
    ("builder", "basic_nodes_and_channels_builder")
])
def basic_nodes_and_channels(request):
    """Parameterized fixture providing basic graph with barrier synchronization using both construction methods."""
    _, fixture_name = request.param
    return request.getfixturevalue(fixture_name)


@pytest.fixture
def conditional_routing_direct():
    """Fixture providing conditional routing graph using direct construction."""

    def pick_target():
        return "D"

    def fn_int():
        return 42

    def fn_receive():
        return "received"

    # Define channels
    channels = [
        TriggerChannel("A"),
        TriggerChannel("D"),
        TriggerChannel("E")
    ]

    # Define nodes and routers
    nodes = {
        "A": PregelNode("A", fn_int, [ConditionalRouter(selector=pick_target)]),
        "D": PregelNode("D", fn_receive, [StaticRouter([])]),
        "E": PregelNode("E", fn_receive, [StaticRouter([])]),
    }

    return nodes, channels


@pytest.fixture
def conditional_routing_builder():
    """Fixture providing conditional routing graph using builder."""

    def pick_target():
        return "D"

    def fn_int(): return 42

    def fn_receive(): return "received"

    builder = PregelGraphBuilder()
    builder.add_node("A", fn_int)
    builder.add_node("D", fn_receive)
    builder.add_node("E", fn_receive)
    builder.add_branch("A", pick_target)

    return builder.nodes, builder.channels


@pytest.fixture(params=[
    ("direct", "conditional_routing_direct"),
    ("builder", "conditional_routing_builder")
])
def conditional_routing(request):
    """Parameterized fixture providing conditional routing graph using both construction methods."""
    _, fixture_name = request.param
    return request.getfixturevalue(fixture_name)


@pytest.fixture
def multi_routing_direct():
    """Fixture providing multi-routing graph using direct construction."""

    # Define node functions
    def fn_int():
        return 1

    def fn_receive():
        return 2

    def fn_end():
        print("END")

    def pick_target():
        # Simple condition: if value > 0, go to E, otherwise go to F
        return "E"

    # Define barrier channels
    abc_to_d = BarrierChannel("D", expected={"A", "B", "C"})
    ay_to_d = BarrierChannel("D", expected={"A", "Y"})
    deg_to_end = BarrierChannel("END", expected={"D", "E", "G"})

    # Define channels
    channels = [
        TriggerChannel("START"),
        TriggerChannel("A"),
        TriggerChannel("B"),
        TriggerChannel("C"),
        TriggerChannel("X"),
        TriggerChannel("Y"),
        TriggerChannel("E"),
        TriggerChannel("F"),
        TriggerChannel("G"),
        TriggerChannel("D"),  # For Y -> D
        TriggerChannel("END"),
        abc_to_d,  # Barrier A,B,C -> D
        ay_to_d,  # Barrier A,Y -> D
        deg_to_end,  # Barrier D,E,G -> END
    ]

    # Define nodes
    nodes = {
        # START → A,B,C,X
        "START": PregelNode("START", fn_int, [
            StaticRouter(["A", "B", "C", "X"])
        ]),

        # A routes to:
        # - G (static)
        # - E or F (conditional)
        # - D (barrier A,B,C)
        # - D (barrier A,X)
        "A": PregelNode("A", fn_int, [
            StaticRouter(["G"]),
            ConditionalRouter(selector=pick_target),
            BarrierRouter([abc_to_d.key, ay_to_d.key]),
        ]),

        "B": PregelNode("B", fn_int, [
            BarrierRouter([abc_to_d.key]),
        ]),

        "C": PregelNode("C", fn_int, [
            BarrierRouter([abc_to_d.key]),
        ]),

        # X routes to:
        # - Y (static)
        # - D (barrier A,X)
        "X": PregelNode("X", fn_int, [
            StaticRouter(["Y"]),
            BarrierRouter([ay_to_d.key]),
        ]),

        # Y -> D (trigger channel)
        "Y": PregelNode("Y", fn_receive, [
            StaticRouter(["D"])
        ]),

        # D triggered either by barrier(A,B,C) or by barrier(A,X) or by Y
        "D": PregelNode("D", fn_receive, [
            BarrierRouter([deg_to_end.key]),
        ]),

        "E": PregelNode("E", fn_receive, [
            BarrierRouter([deg_to_end.key]),
        ]),

        "F": PregelNode("F", fn_receive, [
            StaticRouter(["END"]),
        ]),

        "G": PregelNode("G", fn_receive, [
            BarrierRouter([deg_to_end.key]),
        ]),

        "END": PregelNode("END", fn_end, [
            StaticRouter([]),
        ]),
    }

    return nodes, channels


@pytest.fixture
def multi_routing_builder():
    """Fixture providing multi-routing graph using builder."""

    # Define node functions
    def fn_int(): return 1

    def fn_receive(): return 2

    def fn_end(): print("END")

    def pick_target(): return "E"

    builder = PregelGraphBuilder()

    builder.add_node("START", fn_int)
    builder.add_node("A", fn_int)
    builder.add_node("B", fn_int)
    builder.add_node("C", fn_int)
    builder.add_node("X", fn_int)
    builder.add_node("Y", fn_receive)
    builder.add_node("D", fn_receive)
    builder.add_node("E", fn_receive)
    builder.add_node("F", fn_receive)
    builder.add_node("G", fn_receive)
    builder.add_node("END", fn_end)

    builder.add_edge("START", ["A", "B", "C", "X"])  # fan-out
    builder.add_edge("A", "G")  # static
    builder.add_branch("A", pick_target)  # conditional
    builder.add_edge(["A", "B", "C"], "D")  # barrier A,B,C -> D
    builder.add_edge(["A", "Y"], "D")  # barrier A,Y -> D
    builder.add_edge("X", "Y")  # static
    builder.add_edge("Y", "D")  # static
    builder.add_edge(["D", "E", "G"], "END")  # barrier D,E,G -> END
    builder.add_edge("F", "END")  # static

    return builder.nodes, builder.channels


@pytest.fixture(params=[
    ("direct", "multi_routing_direct"),
    ("builder", "multi_routing_builder")
])
def multi_routing(request):
    """Parameterized fixture providing multi-routing graph using both construction methods."""
    _, fixture_name = request.param
    return request.getfixturevalue(fixture_name)


@pytest.fixture
def nested_subgraph_builder():
    """Fixture providing a graph with a nested subgraph, demonstrating
    FIRST_EXCEPTION resume behavior.

    Outer Graph: start -> a -> end

    Inner Subgraph (node a):
    start1 -> [a1, a2, a3] ()
    a1 (0.2s, raises RuntimeError)
    a2 (1.0s, cancelled)
    a3 (0s, passes)
    """

    async def fn_a1_fail(config):
        await asyncio.sleep(0.2)
        raise RuntimeError("a1 exception")

    async def fn_a2_slow(config):
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise
        return

    def fn_a3_fast(config):
        return "a3_done"

    def fn_pass():
        return "pass"

    inner_builder = PregelGraphBuilder()

    inner_builder.add_node("start1", fn_pass)
    inner_builder.add_node("a1", fn_a1_fail)
    inner_builder.add_node("a2", fn_a2_slow)
    inner_builder.add_node("a3", fn_a3_fast)
    inner_builder.add_node("end1", fn_pass)

    inner_builder.add_edge("start1", ("a1", "a2", "a3"))

    inner_builder.add_edge(("a1", "a2", "a3"), "end1")

    inner_nodes, inner_channels = inner_builder.nodes, inner_builder.channels

    def inner_logger(loop):
        print(f"[{loop.config['ns']}] Inner Step {loop.step}, Active: {list(loop.active_nodes)}")

    inner_app = Pregel(inner_nodes, inner_channels,
                       checkpointer=default_inmemory_checkpointer.graph_checkpointer(),
                       after_tick=inner_logger)

    class RunInner:
        def __init__(self, inner_app):
            self.inner_app = inner_app

        async def __call__(self, state, config):
            print(f"[{config['ns']}] Subgraph Invoked.")
            return await self.inner_app.ainvoke(config, durability="exit")

    builder = PregelGraphBuilder()
    builder.add_node("start", fn_pass)
    builder.add_node("a", RunInner(inner_app))
    builder.add_node("end", fn_pass)

    builder.add_edge("start", "a")
    builder.add_edge("a", "end")

    return builder.nodes, builder.channels


@pytest.fixture
def linear_nested_subgraph_setup():
    async def fn_a1_fail(config):
        pass

    def fn_generic_pass():
        return

    execution_trace = []

    def inner_logger(loop):
        execution_trace.append({
            "step": loop.step,
            "active_nodes": list(loop.active_nodes),
            "ns": loop.config['ns']
        })
        print(f"[{loop.config['ns']}] Inner Step {loop.step}, Active: {list(loop.active_nodes)}")

    inner_builder = PregelGraphBuilder()
    inner_builder.add_node("start1", fn_generic_pass)
    inner_builder.add_node("a1", fn_a1_fail)
    inner_builder.add_node("b1", fn_generic_pass)
    inner_builder.add_node("c1", fn_generic_pass)
    inner_builder.add_node("end1", fn_generic_pass)

    inner_builder.add_edge("start1", "a1")
    inner_builder.add_edge("a1", "b1")
    inner_builder.add_edge("b1", "c1")
    inner_builder.add_edge("c1", "end1")
    inner_builder.add_edge(START, "start1")
    inner_builder.add_edge("end1", END)

    inner_app = Pregel(
        nodes=inner_builder.nodes,
        channels=inner_builder.channels,
        checkpointer=default_inmemory_checkpointer.graph_checkpointer(),
        after_tick=inner_logger
    )

    class RunInner:
        def __init__(self, inner_app):
            self.inner_app = inner_app

        async def __call__(self, state, config):
            print(f"[{config['ns']}] Subgraph Invoked by C.")
            return await self.inner_app.ainvoke(config, durability="exit")

    outer_builder = PregelGraphBuilder()

    outer_builder.add_node("start", fn_generic_pass)
    outer_builder.add_node("a", fn_generic_pass)
    outer_builder.add_node("b", fn_generic_pass)
    # inner node
    outer_builder.add_node("c", RunInner(inner_app))
    outer_builder.add_node("d", fn_generic_pass)
    outer_builder.add_node("end", fn_generic_pass)

    outer_builder.add_edge("start", "a")
    outer_builder.add_edge("a", "b")
    outer_builder.add_edge("b", "c")
    outer_builder.add_edge("c", "d")
    outer_builder.add_edge("d", "end")
    outer_builder.add_edge(START, "start")
    outer_builder.add_edge("end", END)

    # Checkpointer 实例
    checkpointer = default_inmemory_checkpointer.graph_checkpointer()

    execution_trace = []

    def logger(loop):
        execution_trace.append({
            "step": loop.step,
            "active_nodes": list(loop.active_nodes),
            "ns": loop.config['ns']
        })
        print(f"[Outer] Step {loop.step}, Active: {list(loop.active_nodes)}")

    graph = outer_builder.build(
        checkpointer=checkpointer,
        after_tick=logger)
    return graph, execution_trace


@pytest.mark.asyncio
class TestPregelV2:
    async def test_barrier_wait_for_all(self, basic_nodes_and_channels):
        """Test the barrier synchronization.

        Graph structure:
        start -> a -> a1 --\
                b --------\
                c ---------\-> collect -> end
                d --------/
        """
        nodes, channels = basic_nodes_and_channels

        # Track execution steps and active nodes
        execution_trace = []

        def logger(loop):
            execution_trace.append({
                "step": loop.step,
                "active_nodes": list(loop.active_nodes),
            })

        # Create and run the graph
        app = Pregel(nodes, channels, initial="start", after_tick=logger)
        await app.ainvoke()

        # Verify execution trace
        assert len(execution_trace) == 5

        # Step 0: start node
        assert execution_trace[0]["active_nodes"] == ["start"]

        # Step 1: a, b, c, d nodes
        assert set(execution_trace[1]["active_nodes"]) == {"a", "b", "c", "d"}

        # Step 2: a1 node
        assert execution_trace[2]["active_nodes"] == ["a1"]

        # Step 3: collect node (barrier)
        assert execution_trace[3]["active_nodes"] == ["collect"]

        # Step 4: end node
        assert execution_trace[4]["active_nodes"] == ["end"]

    async def test_conditional_routing(self, conditional_routing):
        """Test conditional router.

        Graph structure:
        A -> D (if output is even)
        A -> E (if output is odd)
        """
        nodes, channels = conditional_routing

        # Track execution steps and active nodes
        execution_trace = []

        def logger(loop):
            execution_trace.append({
                "step": loop.step,
                "active_nodes": list(loop.active_nodes),
            })

        app = Pregel(nodes, channels, initial="A", after_tick=logger)
        await app.ainvoke()

        # Verify execution trace
        assert len(execution_trace) == 2  # 2 steps expected

        # Step 0: A node
        assert execution_trace[0]["active_nodes"] == ["A"]

        # Step 1: D node (since 42 is even)
        assert execution_trace[1]["active_nodes"] == ["D"]

        # Verify E node was not activated (since 42 is even, not odd)
        all_active_nodes = [node for trace in execution_trace for node in trace["active_nodes"]]
        assert "E" not in all_active_nodes

    async def test_multi_routing(self, multi_routing):
        """Test multi-routing.

        a | b | c  --->  d
        a ---> e | f    (conditional)
        a ---> g
        x ---> y ---> d ---> end
        a    | y ---> d
        d | e | g ---> end
        """
        nodes, channels = multi_routing

        # Track execution steps and active nodes
        execution_trace = []

        def logger(loop):
            execution_trace.append({
                "step": loop.step,
                "active_nodes": list(loop.active_nodes),
            })
            print(f"step:{loop.step}, nodes:{list(loop.active_nodes)}")

        # Build Pregel graph
        graph = Pregel(
            nodes=nodes,
            channels=channels,
            initial="START",
            after_tick=logger
        )

        await graph.ainvoke()

        assert len(execution_trace) == 4

        # Step 0: START node
        assert execution_trace[0]["active_nodes"] == ["START"]

        assert set(execution_trace[1]["active_nodes"]) == {'B', 'A', 'C', 'X'}
        assert set(execution_trace[2]["active_nodes"]) == {'Y', 'E', 'G', 'D'}
        assert set(execution_trace[3]["active_nodes"]) == {'END', 'D'}

    async def test_subgraph_with_exception(self, nested_subgraph_builder):
        nodes, channels = nested_subgraph_builder
        execution_trace = []

        def logger(loop):
            execution_trace.append({
                "step": loop.step,
                "active_nodes": list(loop.active_nodes),
                "ns": loop.config['ns']
            })
            print(f"[Outer] Step {loop.step}, Active: {list(loop.active_nodes)}")

        graph = Pregel(
            nodes=nodes,
            channels=channels,
            initial="start",
            checkpointer=default_inmemory_checkpointer.graph_checkpointer(),
            after_tick=logger
        )
        config = PregelConfig(session_id="test_parallel_fail", ns="start-a-end", recursion_limit=10)
        print("\n=============== Invoke 1 (Failure) ===============")

        try:
            await graph.ainvoke(config)
        except RuntimeError as e:
            assert "a1 exception" in str(e)

        print("\n=============== Invoke 2 (Resume) ===============")

        try:
            await graph.ainvoke(config)
        except RuntimeError as e:
            print("second exception:", e)

        assert 'start' not in [d['active_nodes'] for d in execution_trace if
                               d['ns'] == 'start-a-end' and d['step'] > 1]

    async def test_recursion_limit_recovery(self, linear_nested_subgraph_setup):
        """
        Outer Graph: start -> a -> b -> [c (Subgraph)] -> d -> end
        Inner Subgraph (c): start1 -> a1 -> b1| -> c1 -> end1
        """
        graph, execution_trace = linear_nested_subgraph_setup
        RECURSION_LIMIT = 3
        SESSION_ID = "test_recursion_limit_nested"
        NS_OUTER = "outer-linear-test"

        config = PregelConfig(
            session_id=SESSION_ID,
            ns=NS_OUTER,
            recursion_limit=RECURSION_LIMIT
        )

        print("\n=============== Invoke 1 (Failure at Step 4 / Node C) ===============")

        with pytest.raises(RecursionError) as excinfo:
            await graph.ainvoke(config)

        assert f"Recursion limit of {RECURSION_LIMIT} reached" in str(excinfo.value)

        assert execution_trace[-1]['active_nodes'] == ['b']

        checkpoint = await graph.checkpointer.get(SESSION_ID, config['ns'])
        assert checkpoint is not None
        assert checkpoint.step == 4
        assert not checkpoint.pending_node
        print("Channel Values:", checkpoint.channel_values)
        print("pending_buffer:", checkpoint.pending_buffer)
        print("\n=============== Invoke 2 (Resume from Node C) ===============")

        execution_trace.clear()
        with pytest.raises(RecursionError) as excinfo:
            await graph.ainvoke(config)
        assert f"Recursion limit of {RECURSION_LIMIT} reached" in str(excinfo.value)
        assert execution_trace[-1]['active_nodes'] == ['b1']
        checkpoint = await graph.checkpointer.get(SESSION_ID, config['ns'])

        print("\n=============== Invoke 3 (Resume from Node b1) ===============")

        execution_trace.clear()
        await graph.ainvoke(config)
        assert execution_trace[-1]['active_nodes'] == ['end']
