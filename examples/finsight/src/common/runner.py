from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.runner.runner import resource_mgr
from openjiuwen.core.utils.tool.function.function import LocalFunction
from openjiuwen.core.utils.tool.param import Param

from src.tools import get_tool_categories, get_tool_by_name


def get_and_register_local_tools():
    # Attach other API tools
    tool_mgr = resource_mgr.tool()
    tool_dict = {}
    for tool_type, tool_name_list in get_tool_categories().items():
        tool_list = []
        for tool_name in tool_name_list:
            tool_instance = get_tool_by_name(tool_name)()
            params = [
                Param(
                    name=p.get("name"),
                    description=p.get("description"),
                    param_type=p.get("type") if p.get("type") != "List[str]" else "List",
                    required=p.get("required")
                )
                for p in tool_instance.parameters
            ]
            tool = LocalFunction(
                name=tool_instance.name,
                description=tool_instance.description,
                params=params,
                func=tool_instance.api_function
            )
            tool_list.append(tool)
            tool_mgr.add_tool(tool.name, tool)
        tool_dict[tool_type] = tool_list
    return tool_dict

