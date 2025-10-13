

from dataclasses import dataclass
from typing import Optional

from jiuwen.graph.visualization.drawable_graph import DrawableGraph
from jiuwen.graph.visualization.drawable_node import DrawableNode


@dataclass
class DrawableSubgraphNode(DrawableNode):
    subgraph: Optional[DrawableGraph] = None