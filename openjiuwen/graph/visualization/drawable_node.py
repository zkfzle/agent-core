from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class DrawableNode:
    id: str
    name: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None