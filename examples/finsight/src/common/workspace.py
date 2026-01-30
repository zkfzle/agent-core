from typing import List


from src.tools import ToolResult, ClickResult, SearchResult


class FinSightWorkspace:
    collect_results: List[ToolResult] = []
    click_results: List[ClickResult] = []
    search_results: List[SearchResult] = []
    analysis_results: List[ToolResult] = []


# Global logger singleton
_workspace_instance = None


def get_workspace() -> FinSightWorkspace:
    """Return the global Workspace instance."""
    global _workspace_instance
    if _workspace_instance is None:
        _workspace_instance = FinSightWorkspace()
    return _workspace_instance


def set_workspace(workspace) :
    """Return the global Workspace instance."""
    global _workspace_instance
    if _workspace_instance is None:
        _workspace_instance = workspace
    raise RuntimeError("Workspace exists!")