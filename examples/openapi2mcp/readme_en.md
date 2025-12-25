# 一、OpenAPI Specification

The OpenAPI Specification (OAS) is an industry-standard format for describing RESTful APIs. It is usually in the form of YAML or JSON files and defines metadata such as API endpoints, parameters, and request/response formats.
# 二、Agent call OpenAPI
* The MCP protocol provides methods for agents to call tools, but existing APIs cannot be directly called by agents. Open APIs can be converted into MCP-Tools, which are then invoked by the agent.
* Implementation Method:
  1. First, obtain the OAS file.
  2. Start a custom MCP_client on the agent side.
  3. Implement a tool_manager in the custom MCP_client to parse the OAS file, converting each API operation into an independent MCP_tool object for management. Each tool object records the corresponding HTTP call method.
  4. Expose a standardized MCP interface to the agent, allowing the agent to use methods such as list_tools and call_tool.

*Below is an executable example in Linux.
## 2.1 Start an OpenAPI service
```python
# fastapi_server.py
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel


class Item(BaseModel):
    id: int
    name: str
    price: float

class ItemCreate(BaseModel):
    name: str
    price: float

class Message(BaseModel):
    message: str

# ---------------FastAPI demo---------------
app = FastAPI(
    title="ItemShop API",
    version="1.0",
    description="An item shop API, include post and get",
    openapi_url="/openapi.json", # we can get openapi.json by "curl http://0.0.0.0:8000/openapi.json"
    docs_url="/docs",
    redoc_url="/redoc",
)

fake_db: List[Item] = [Item(id=1, name='Keyboard', price=99.9)]

@app.get(
    "/items",
    response_model=List[Item],
    summary="list all items",
    description="list all items",
    operation_id="list_items",
    tags=["items"],
)
def list_items():
    return fake_db

@app.get(
    "/items/{item_id}",
    response_model=Item,
    responses={
        404:{"model": Item, "description": "item not found"},
    },
    summary="get an item",
    operation_id="get_item",
    tags=["items"],
)
def get_item(item_id: int):
    for it in fake_db:
        if it.id == item_id:
            return it
    raise HTTPException(status_code=404, detail="item not found")

@app.post(
    "/items",
    response_model=Item,
    status_code=status.HTTP_201_CREATED,
    summary="create a new item",
    operation_id="create_item",
    tags=["item"],
)
def create_item(body: ItemCreate):
    new_id = max((it.id for it in fake_db),default=0)+1
    new_item = Item(id=new_id, name=body.name, price=body.price)
    fake_db.append(new_item)
    return new_item

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
Running `Python fastapi_server.py` will deploy a service on port 8000 locally by default.

## 2.2 获取OAS文件
* Executing `curl http://0.0.0.0:8000/openapi.json` will retrieve the OAS file for this service.
* Note that the PATH field in the original OAS file is missing the IP address and port number. You need to complete `http://0.0.0.0:8000/items` to create a fully functional URL.

## 2.3 client执行转换并调用OpenAPI
```python
import asyncio
from openjiuwen.core.utils.tool.mcp.openapi_client import OpenApiClient


async def main():
    client = OpenApiClient(["./openapi.json"], "test")
    # Execute conversion process
    _ = await client.connect()

    tools = await client.list_tools()
    print("Available tools:", [t.name for t in tools])

    items = await client.call_tool("list_items",{})
    print("list_items", items)
    
    _ = client.disconnect()

asyncio.run(main())
```
Executing the above script will show that the three API operations in the sample service have been converted into tool objects, and the execution results of the APIs can be obtained using the call_tool method.

## 2.4 The conversion is achieved using jiuwen's tool_manager.
```python
import asyncio
from openjiuwen.core.runtime.resources_manager.tool_manager import ToolMgr
from openjiuwen.core.utils.tool.mcp.base import ToolServerConfig

async def main():
    mgr = ToolMgr()
    # The client_type is specified as "openapi", and the path to the OAS file is passed in the params.
    cfg = ToolServerConfig(server_name="api", client_type="openapi", server_path=["openapi.json"])
    _ = await mgr.add_tool_servers([cfg])
    
    for info in mgr.get_tool_infos():
        print(info.dict())

asyncio.run(main())
```
Executing the above script shows that the OpenAPI tool object is also obtained in jiuwen's ToolMgr.