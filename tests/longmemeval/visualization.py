#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
图记忆可视化脚本

用于可视化 evaluate.py 中构建的图记忆，包括实体、关系和事件。
使用 Mermaid 格式进行可视化，参考仓库中已有的可视化方法。
"""

import os
import sys
import argparse
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openjiuwen.core.memory.store.graph_store.base import GraphMemory
from openjiuwen.core.memory.config.graph.config import GraphConfig, LLMConfig
from openjiuwen.core.memory.config.graph import query_expr

try:
    from mermaid import Mermaid
    from mermaid.flowchart import FlowChart, Link, LinkShape, Node, Direction
    _MERMAID_AVAILABLE = True
except ImportError:
    _MERMAID_AVAILABLE = False


class GraphMemoryVisualizer:
    """图记忆可视化器
    
    从 GraphMemory 中提取实体、关系和事件，并使用 Mermaid 格式进行可视化。
    """
    
    def __init__(self, graph_memory: GraphMemory, user_id: Optional[str] = None):
        """
        初始化可视化器
        
        Args:
            graph_memory: 图记忆实例
            user_id: 用户ID，如果提供则只可视化该用户的数据
        """
        self.graph_memory = graph_memory
        self.user_id = user_id
        self.entities: Dict[str, Dict] = {}
        self.relations: Dict[str, Dict] = {}
        self.episodes: Dict[str, Dict] = {}
    
    def load_data(self, limit: Optional[int] = None):
        """从图记忆中加载所有数据"""
        print("正在从图记忆中加载数据...")
        
        # 构建查询表达式
        expr = None
        if self.user_id:
            expr = query_expr.filter_user(self.user_id)
        
        # 如果没有提供 limit，使用一个较大的默认值
        query_limit = limit if limit is not None else 10000
        
        # 查询实体
        print("  查询实体...")
        entities_data = self.graph_memory.db_backend.query(
            collection="entities",
            expr=expr,
            limit=query_limit
        )
        for entity in entities_data:
            self.entities[entity.get("uuid", "")] = entity
        print(f"  找到 {len(self.entities)} 个实体")
        
        # 查询关系
        print("  查询关系...")
        relations_data = self.graph_memory.db_backend.query(
            collection="relations",
            expr=expr,
            limit=query_limit
        )
        for relation in relations_data:
            self.relations[relation.get("uuid", "")] = relation
        print(f"  找到 {len(self.relations)} 个关系")
        
        # 查询事件
        print("  查询事件...")
        episodes_data = self.graph_memory.db_backend.query(
            collection="episodes",
            expr=expr,
            limit=query_limit
        )
        for episode in episodes_data:
            self.episodes[episode.get("uuid", "")] = episode
        print(f"  找到 {len(self.episodes)} 个事件")
        
        print(f"数据加载完成: {len(self.entities)} 个实体, {len(self.relations)} 个关系, {len(self.episodes)} 个事件")
    
    def _sanitize_node_id(self, node_id: str) -> str:
        """清理节点ID，使其适合 Mermaid 格式"""
        # Mermaid 节点ID只能包含字母、数字、下划线和连字符
        import re
        # 替换所有非字母数字字符为下划线
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', node_id)
        # 确保不以数字开头
        if sanitized and sanitized[0].isdigit():
            sanitized = 'n' + sanitized
        # 限制长度
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        return sanitized
    
    def _format_node_label(self, name: Optional[str], content: Optional[str] = None, max_length: int = 30) -> str:
        """格式化节点标签"""
        label = name or "未命名"
        if content:
            # 添加内容摘要
            content_preview = content[:max_length] + "..." if len(content) > max_length else content
            label = f"{label}\\n({content_preview})"
        # 转义特殊字符
        label = label.replace('"', '\\"').replace('\n', '\\n')
        return label
    
    def to_mermaid(self, title: str = "图记忆可视化", show_episodes: bool = True, max_nodes: Optional[int] = None) -> str:
        """将图记忆转换为 Mermaid 格式
        
        Args:
            title: 图表标题
            show_episodes: 是否显示事件节点
            max_nodes: 最大节点数量（用于限制大型图）
        
        Returns:
            Mermaid 格式的字符串
        """
        if not _MERMAID_AVAILABLE:
            return self._to_text_format()
        
        # 收集所有相关的实体（有关系的实体）
        relevant_entities: Set[str] = set()
        for relation in self.relations.values():
            lhs = relation.get("lhs")
            rhs = relation.get("rhs")
            if lhs:
                relevant_entities.add(lhs if isinstance(lhs, str) else str(lhs))
            if rhs:
                relevant_entities.add(rhs if isinstance(rhs, str) else str(rhs))
        
        # 如果显示事件，也收集与事件相关的实体
        if show_episodes:
            for episode in self.episodes.values():
                entities = episode.get("entities", [])
                relevant_entities.update(entities)
        
        # 限制节点数量
        if max_nodes and len(relevant_entities) > max_nodes:
            print(f"警告: 节点数量 ({len(relevant_entities)}) 超过限制 ({max_nodes})，将只显示部分节点")
            relevant_entities = set(list(relevant_entities)[:max_nodes])
        
        # 创建节点映射
        mermaid_nodes: Dict[str, Node] = {}
        node_id_map: Dict[str, str] = {}  # 原始UUID到Mermaid节点ID的映射
        used_node_ids: Set[str] = set()  # 已使用的节点ID集合
        
        # 为实体创建节点
        for entity_uuid in relevant_entities:
            entity = self.entities.get(entity_uuid)
            if not entity:
                continue
            
            # 生成唯一的节点ID
            base_id = self._sanitize_node_id(entity_uuid[:20])  # 使用UUID的前20个字符
            node_id = base_id
            counter = 1
            while node_id in used_node_ids:
                node_id = f"{base_id}_{counter}"
                counter += 1
            used_node_ids.add(node_id)
            node_id_map[entity_uuid] = node_id
            
            name = entity.get("name", "未命名实体")
            content = entity.get("content", "")
            label = self._format_node_label(name, content, max_length=20)
            
            mermaid_nodes[node_id] = Node(
                id_=node_id,
                content=label,
                shape="round-edge"  # 使用圆角矩形表示实体
            )
        
        # 如果显示事件，为事件创建节点
        episode_nodes: Dict[str, Node] = {}
        episode_id_map: Dict[str, str] = {}  # 事件UUID到节点ID的映射
        if show_episodes:
            for episode_uuid, episode in list(self.episodes.items())[:50]:  # 限制事件数量
                # 生成唯一的事件节点ID
                base_id = self._sanitize_node_id(f"ep_{episode_uuid[:15]}")
                node_id = base_id
                counter = 1
                while node_id in used_node_ids:
                    node_id = f"{base_id}_{counter}"
                    counter += 1
                used_node_ids.add(node_id)
                episode_id_map[episode_uuid] = node_id
                
                content = episode.get("content", "无内容")
                label = self._format_node_label("事件", content, max_length=25)
                
                episode_nodes[node_id] = Node(
                    id_=node_id,
                    content=label,
                    shape="normal"  # 使用普通矩形表示事件
                )
        
        # 创建边（关系）
        links: List[Link] = []
        for relation in self.relations.values():
            lhs = relation.get("lhs")
            rhs = relation.get("rhs")
            
            # 获取实体UUID
            lhs_uuid = lhs if isinstance(lhs, str) else (getattr(lhs, 'uuid', None) if hasattr(lhs, 'uuid') else str(lhs))
            rhs_uuid = rhs if isinstance(rhs, str) else (getattr(rhs, 'uuid', None) if hasattr(rhs, 'uuid') else str(rhs))
            
            if not lhs_uuid or not rhs_uuid:
                continue
            
            if lhs_uuid not in node_id_map or rhs_uuid not in node_id_map:
                continue
            
            source_id = node_id_map[lhs_uuid]
            target_id = node_id_map[rhs_uuid]
            
            # 关系名称作为边的标签
            relation_name = relation.get("name", "")
            relation_content = relation.get("content", "")
            message = relation_name or relation_content[:20] or ""
            if message:
                message = f'"{message[:30]}"'  # 限制长度并转义
            
            links.append(Link(
                origin=mermaid_nodes[source_id],
                end=mermaid_nodes[target_id],
                shape=LinkShape.NORMAL,
                message=message
            ))
        
        # 如果显示事件，创建实体到事件的边
        if show_episodes:
            for episode_uuid, episode in list(self.episodes.items())[:50]:
                if episode_uuid not in episode_id_map:
                    continue
                
                episode_node_id = episode_id_map[episode_uuid]
                if episode_node_id not in episode_nodes:
                    continue
                
                episode_entities = episode.get("entities", [])
                for entity_uuid in episode_entities:
                    if entity_uuid in node_id_map:
                        source_id = node_id_map[entity_uuid]
                        links.append(Link(
                            origin=mermaid_nodes[source_id],
                            end=episode_nodes[episode_node_id],
                            shape=LinkShape.DOTTED,  # 使用虚线表示实体-事件关系
                            message='"提及"'
                        ))
        
        # 合并所有节点
        all_nodes = list(mermaid_nodes.values()) + list(episode_nodes.values())
        
        # 创建流程图
        chart = FlowChart(title, all_nodes, links)
        return chart.script
    
    def _to_text_format(self) -> str:
        """当 Mermaid 不可用时，使用文本格式输出"""
        lines = ["图记忆可视化（文本格式）", "=" * 60]
        
        lines.append(f"\n实体 ({len(self.entities)} 个):")
        for i, (uuid, entity) in enumerate(list(self.entities.items())[:20], 1):
            name = entity.get("name", "未命名")
            content = entity.get("content", "")[:50]
            lines.append(f"  {i}. {name} ({uuid[:20]}...)")
            if content:
                lines.append(f"     内容: {content}...")
        
        lines.append(f"\n关系 ({len(self.relations)} 个):")
        for i, (uuid, relation) in enumerate(list(self.relations.items())[:20], 1):
            name = relation.get("name", "未命名关系")
            lhs = relation.get("lhs", "?")
            rhs = relation.get("rhs", "?")
            lhs_str = lhs if isinstance(lhs, str) else str(lhs)[:20]
            rhs_str = rhs if isinstance(rhs, str) else str(rhs)[:20]
            lines.append(f"  {i}. {lhs_str} --[{name}]--> {rhs_str}")
        
        lines.append(f"\n事件 ({len(self.episodes)} 个):")
        for i, (uuid, episode) in enumerate(list(self.episodes.items())[:20], 1):
            content = episode.get("content", "")[:50]
            lines.append(f"  {i}. {uuid[:20]}...: {content}...")
        
        return "\n".join(lines)
    
    def to_mermaid_png(self, title: str = "图记忆可视化", 
                       show_episodes: bool = True, max_nodes: Optional[int] = None) -> bytes:
        """将图记忆转换为 Mermaid PNG 图片
        
        Args:
            title: 图表标题
            show_episodes: 是否显示事件节点
            max_nodes: 最大节点数量
        
        Returns:
            PNG 图片的字节数据
        """
        if not _MERMAID_AVAILABLE:
            raise ImportError("mermaid-py 未安装，无法生成 PNG。安装命令: pip install mermaid-py")
        
        mermaid_code = self.to_mermaid(title, show_episodes, max_nodes)
        mermaid = Mermaid(mermaid_code)
        return mermaid.img_response.content
    
    def to_mermaid_svg(self, title: str = "图记忆可视化",
                       show_episodes: bool = True, max_nodes: Optional[int] = None) -> bytes:
        """将图记忆转换为 Mermaid SVG 图片
        
        Args:
            title: 图表标题
            show_episodes: 是否显示事件节点
            max_nodes: 最大节点数量
        
        Returns:
            SVG 图片的字节数据
        """
        if not _MERMAID_AVAILABLE:
            raise ImportError("mermaid-py 未安装，无法生成 SVG。安装命令: pip install mermaid-py")
        
        mermaid_code = self.to_mermaid(title, show_episodes, max_nodes)
        mermaid = Mermaid(mermaid_code)
        return mermaid.svg_response.content
    
    def save_mermaid(self, output_path: str, title: str = "图记忆可视化", 
                     show_episodes: bool = True, max_nodes: Optional[int] = None):
        """保存 Mermaid 代码到文件"""
        mermaid_code = self.to_mermaid(title, show_episodes, max_nodes)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(mermaid_code)
        print(f"Mermaid 代码已保存到: {output_path}")
    
    def save_png(self, output_path: str, title: str = "图记忆可视化",
                 show_episodes: bool = True, max_nodes: Optional[int] = None):
        """保存 PNG 图片到文件"""
        png_data = self.to_mermaid_png(title, output_path, show_episodes, max_nodes)
        with open(output_path, 'wb') as f:
            f.write(png_data)
        print(f"PNG 图片已保存到: {output_path}")
    
    def save_svg(self, output_path: str, title: str = "图记忆可视化",
                 show_episodes: bool = True, max_nodes: Optional[int] = None):
        """保存 SVG 图片到文件"""
        svg_data = self.to_mermaid_svg(title, output_path, show_episodes, max_nodes)
        with open(output_path, 'wb') as f:
            f.write(svg_data)
        print(f"SVG 图片已保存到: {output_path}")


def init_graph_memory(
    storage_path: Optional[str] = None,
    language: str = "cn",
    llm_config: Optional[LLMConfig] = None
) -> GraphMemory:
    """初始化图记忆（与 evaluate.py 中的函数相同）"""
    import os
    from dotenv import load_dotenv
    
    # 加载环境变量
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    
    # 配置图记忆数据库
    local_endpoint = os.getenv("LOCAL_ENDPOINT")
    db_uri = f"http://{local_endpoint}"
    print(f"db_uri: {db_uri}")
    
    db_config = GraphConfig(
        uri=db_uri,
        name="evaluation",
        backend="milvus",
        wipe_at_startup=False,
        timeout=5.0,
    )
    
    # 配置 LLM
    if llm_config is None:
        try:
            llm_config = LLMConfig.default_config()
            print("从环境变量读取到LLM 配置")
        except (ValueError, KeyError) as e:
            print(f"⚠️  无法从环境变量读取 LLM 配置: {e}")
            print("请设置 JIUWEN_GRAPH_MEM_LLM_URL, JIUWEN_GRAPH_MEM_LLM_KEY, JIUWEN_GRAPH_MEM_LLM_MODEL")
            raise
    
    # 创建图记忆实例
    graph_memory = GraphMemory(
        db_config=db_config,
        llm_config=llm_config,
        language=language,
    )
    
    print("✅ 图记忆初始化成功")
    return graph_memory


async def main():
    parser = argparse.ArgumentParser(description="图记忆可视化脚本")
    parser.add_argument("--user_id", type=str, default=None,
                        help="用户ID，如果提供则只可视化该用户的数据")
    parser.add_argument("--output", type=str, default="graph_memory",
                        help="输出文件路径（不含扩展名，默认: graph_memory）")
    parser.add_argument("--format", type=str, choices=["mermaid", "png", "svg", "all"], default="all",
                        help="输出格式 (默认: all)")
    parser.add_argument("--title", type=str, default="图记忆可视化",
                        help="图表标题 (默认: 图记忆可视化)")
    parser.add_argument("--show_episodes", action="store_true", default=True,
                        help="是否显示事件节点 (默认: True)")
    parser.add_argument("--hide_episodes", action="store_true",
                        help="隐藏事件节点")
    parser.add_argument("--max_nodes", type=int, default=None,
                        help="最大节点数量（用于限制大型图）")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制查询的数据数量（用于快速测试）")
    parser.add_argument("--language", type=str, choices=["cn", "en"], default="cn",
                        help="图记忆语言 (默认: cn)")
    
    args = parser.parse_args()
    
    # 初始化图记忆
    print("初始化图记忆...")
    graph_memory = init_graph_memory(language=args.language)
    
    # 创建可视化器
    visualizer = GraphMemoryVisualizer(graph_memory, user_id=args.user_id)
    
    # 加载数据
    visualizer.load_data(limit=args.limit)
    
    # 确定是否显示事件
    show_episodes = args.show_episodes and not args.hide_episodes
    
    # 生成可视化
    if args.format in ["mermaid", "all"]:
        visualizer.save_mermaid(
            f"{args.output}.mmd",
            title=args.title,
            show_episodes=show_episodes,
            max_nodes=args.max_nodes
        )
    
    if args.format in ["png", "all"] and _MERMAID_AVAILABLE:
        try:
            visualizer.save_png(
                f"{args.output}.png",
                title=args.title,
                show_episodes=show_episodes,
                max_nodes=args.max_nodes
            )
        except Exception as e:
            print(f"⚠️  生成 PNG 失败: {e}")
            import traceback
            traceback.print_exc()
    
    if args.format in ["svg", "all"] and _MERMAID_AVAILABLE:
        try:
            visualizer.save_svg(
                f"{args.output}.svg",
                title=args.title,
                show_episodes=show_episodes,
                max_nodes=args.max_nodes
            )
        except Exception as e:
            print(f"⚠️  生成 SVG 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n可视化完成")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

