# 一、OpenAPI规范
OpenAPI规范(OAS,OpenAPI Specification)是用于描述Restful API的行业标准格式，通常以Yaml或Json文件形式存在，它定义了API的端点、参数、请求/响应格式等元数据。

# 二、Agent调用OpenAPI
* MCP协议提供了Agent调用Tool的方法，但是现有存量的API无法直接由Agent进行调用，可以将OpenAPI转换成MCP-Tool，再由Agent实现调用
* 实现方法:
  1. 首先获取OAS文件
  2. 在Agent侧启动一个自定义MCP_client
  3. 自定义MCP_client中实现一个tool_manager，对OAS文件进行解析，将每个API操作都转换成一个独立的MCP_tool对象进行管理，每个tool对象自身记录着相应的http调用方法
  4. 自定义MCP_client对外暴露规范的MCP接口，Agent可以正常的使用list_tools\call_tool等方法

* 下面是一个在Linux中可执行的样例
## 2.1 启动一个OpenAPI服务
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
执行`Python fastapi_server.py` ，默认会在本地8000端口部署一个服务

## 2.2 获取OAS文件
* 执行`curl http://0.0.0.0:8000/openapi.json` 可以获得这个服务的OAS文件。
* 注意原始OAS文件中的PATH字段缺少IP+Port，需要补齐`http://0.0.0.0:8000/items` 变成一个可调用的完整URL。

## 2.3 client执行转换并调用OpenAPI
```python
import asyncio
from openjiuwen.core.utils.tool.mcp.openapi_client import OpenApiClient


async def main():
    client = OpenApiClient(["./openapi.json"], "test")
    # 执行转换流程
    _ = await client.connect()

    tools = await client.list_tools()
    print("Available tools:", [t.name for t in tools])

    items = await client.call_tool("list_items",{})
    print("list_items", items)
    
    _ = client.disconnect()

asyncio.run(main())
```
执行上述脚本，可以看到样例服务中的3个api操作都被转换成立tool对象，并且可以使用call_tool方法获取api的执行结果。

## 2.4 通过jiuwen的tool_manager实现转换
```python
import asyncio
from openjiuwen.core.runtime.resources_manager.tool_manager import ToolMgr
from openjiuwen.core.utils.tool.mcp.base import ToolServerConfig

async def main():
    mgr = ToolMgr()
    # client_type指定"openapi"，在params中传入OAS文件路径
    cfg = ToolServerConfig(server_name="api", client_type="openapi", server_path=["openapi.json"])
    _ = await mgr.add_tool_servers([cfg])
    
    for info in mgr.get_tool_infos():
        print(info.dict())

asyncio.run(main())
```
执行上述脚本，可以看到在jiuwen的ToolMgr中也获得了OpenAPI的tool对象