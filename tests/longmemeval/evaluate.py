#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
LongMemEval 图记忆评估脚本

用于评估图记忆（Graph Memory）在 LongMemEval 数据集上的表现。
图记忆会存储背景信息和对话历史，并在回答问题时检索相关记忆。
"""

import os
import sys
import asyncio
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import socket



from dotenv import load_dotenv
# 加载当前目录的 .env 文件
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)


# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.longmemeval.loader import LongMemEvalLoader, LongMemEvalItem
from openjiuwen.core.memory.store.graph_store.base import GraphMemory
from openjiuwen.core.memory.config.graph.config import GraphConfig, LLMConfig, EpisodeType
from openjiuwen.core.memory.config.graph.extraction_strategies import DEFAULT_STRATEGY
from openjiuwen.core.utils.llm.base import BaseModelInfo
from openjiuwen.core.utils.llm.messages import HumanMessage, SystemMessage
from openjiuwen.core.utils.llm.model_utils.model_factory import ModelFactory


@dataclass
class EvaluationResult:
    """单个测试用例的评估结果"""
    item_id: str
    question: str
    expected_answer: str
    predicted_answer: str
    question_type: str
    is_correct: bool
    evaluation_reason: Optional[str] = None  # LLM 评估的理由
    error: Optional[str] = None
    retrieved_memories: Optional[Dict[str, Any]] = None  # 检索到的记忆信息


@dataclass
class EvaluationMetrics:
    """评估指标"""
    total: int
    correct: int
    accuracy: float
    by_type: Dict[str, Dict[str, float]]  # 按问题类型分组的指标


class GraphMemoryEvaluator:
    """图记忆评估器
    
    使用图记忆存储背景信息和对话历史，并在回答问题时检索相关记忆。
    """
    
    def __init__(
        self,
        graph_memory: GraphMemory,
        llm_model: Any,
        use_background: bool = True,
        use_conversation: bool = True,
        use_llm_evaluator: bool = False,
        evaluator_model: Any = None,
        search_top_k: int = 5,
    ):
        """
        初始化评估器
        
        Args:
            graph_memory: 图记忆实例
            llm_model: 用于回答问题的 LLM 模型
            use_background: 是否使用背景文本
            use_conversation: 是否使用对话历史
            use_llm_evaluator: 是否使用 LLM 评估器
            evaluator_model: LLM 评估器使用的模型（如果 use_llm_evaluator=True）
            search_top_k: 检索记忆时的 top_k 数量
        """
        self.graph_memory = graph_memory
        self.llm_model = llm_model
        self.use_background = use_background
        self.use_conversation = use_conversation
        self.use_llm_evaluator = use_llm_evaluator
        self.evaluator_model = evaluator_model
        self.search_top_k = search_top_k
        
        if use_llm_evaluator and evaluator_model is None:
            raise ValueError("使用 LLM 评估器时必须提供 evaluator_model")
    
    def _format_conversation_history(self, conversations: List) -> str:
        """格式化对话历史为字符串"""
        if not conversations:
            return ""
        
        formatted = []
        for turn in conversations:
            role_name = "用户" if turn.role == "user" else "助手"
            formatted.append(f"{role_name}：{turn.content}")
        
        return "\n".join(formatted)
    
    def _format_retrieved_memories(self, search_results: Dict[str, List[Tuple[float, Any]]]) -> str:
        """格式化检索到的记忆为字符串"""
        if not search_results:
            return ""
        
        formatted_parts = []
        
        # 格式化实体
        if "entity" in search_results and search_results["entity"]:
            formatted_parts.append("相关实体：")
            for score, entity in search_results["entity"][:self.search_top_k]:
                entity_info = f"- {entity.name}"
                if entity.content:
                    entity_info += f": {entity.content}"
                if hasattr(entity, 'attributes') and entity.attributes:
                    attrs = ", ".join([f"{k}={v}" for k, v in entity.attributes.items() if v])
                    if attrs:
                        entity_info += f" ({attrs})"
                formatted_parts.append(entity_info)
        
        # 格式化关系
        if "relation" in search_results and search_results["relation"]:
            formatted_parts.append("\n相关关系：")
            for score, relation in search_results["relation"][:self.search_top_k]:
                relation_info = f"- {relation.name}"
                if relation.content:
                    relation_info += f": {relation.content}"
                formatted_parts.append(relation_info)
        
        # 格式化事件
        if "episode" in search_results and search_results["episode"]:
            formatted_parts.append("\n相关事件：")
            for score, episode in search_results["episode"][:self.search_top_k]:
                episode_info = f"- {episode.content}"
                formatted_parts.append(episode_info)
        
        return "\n".join(formatted_parts) if formatted_parts else ""
    
    async def _add_memories(self, item: LongMemEvalItem, user_id: str):
        """将背景信息和对话历史添加到图记忆"""
        # 添加背景信息
        if self.use_background and item.background_text:
            try:
                print(f"\n📝 添加背景信息...")
                print(f"   User ID: {user_id}")
                print(f"   背景文本前100字: {item.background_text[:100]}...")
                
                result = self.graph_memory.add_memory(
                    src_type=EpisodeType.document,
                    user_id=user_id,
                    content=item.background_text,
                )
                
                print(f"   提取结果: {result}")
                
                self.graph_memory.db_backend.refresh()
                await asyncio.sleep(2)  # 增加等待时间，确保 Milvus 数据可搜索
                
                # 尝试检索，打印详细结果
                test_search = self.graph_memory.search(
                    query=item.background_text[:50],
                    user_id=user_id,
                )
                entity_count = len(test_search.get('entity', []))
                episode_count = len(test_search.get('episode', []))
                print(f"   立即检索验证: entity={entity_count}, episode={episode_count}")
                
                # 如果没找到episode，打印详细信息用于调试
                if episode_count == 0 and result[0].added_episode:
                    print(f"   ⚠️  警告: 添加了 {len(result[0].added_episode)} 个episode，但检索时未找到")
                    print(f"   尝试的查询: {item.background_text[:50]}")
                    # 尝试直接查询数据库验证数据是否存在
                    try:
                        from openjiuwen.core.memory.config.graph import query_expr
                        query_result = self.graph_memory.db_backend.query(
                            collection="episodes",
                            expr=query_expr.filter_user(user_id),
                            limit=10
                        )
                        print(f"   数据库直接查询结果: 找到 {len(query_result)} 个episode（不经过相似度搜索）")
                        if query_result:
                            print(f"   第一个episode内容前100字: {query_result[0].get('content', '')[:100]}")
                    except Exception as e:
                        print(f"   直接查询失败: {e}")
                
            except Exception as e:
                print(f"❌ 添加背景信息失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 添加对话历史（作为对话）
        if self.use_conversation and item.conversations:
            # 将对话历史转换为消息列表格式
            messages = []
            for turn in item.conversations:
                messages.append({
                    "role": turn.role,
                    "content": turn.content
                })

            if messages:
                try:
                    print(f"\n📝 添加对话历史...")
                    print(f"   User ID: {user_id}")
                    print(f"   对话轮数: {len(messages)}")
                    print(f"   第一轮内容前100字: {messages[0]['content'][:100]}...")
                    
                    result = self.graph_memory.add_memory(
                        src_type=EpisodeType.conversation,
                        user_id=user_id,
                        content=messages,
                    )
                    
                    print(f"   提取结果: {result}")
                    
                    self.graph_memory.db_backend.refresh()
                    await asyncio.sleep(2)  # 增加等待时间，确保 Milvus 数据可搜索
                    
                    # 尝试检索，打印详细结果
                    test_search = self.graph_memory.search(
                        query=messages[0]['content'][:50],
                        user_id=user_id,
                    )
                    entity_count = len(test_search.get('entity', []))
                    episode_count = len(test_search.get('episode', []))
                    print(f"   立即检索验证: entity={entity_count}, episode={episode_count}")
                    
                    # 如果没找到episode，打印详细信息用于调试
                    if episode_count == 0 and result[0].added_episode:
                        print(f"   ⚠️  警告: 添加了 {len(result[0].added_episode)} 个episode，但检索时未找到")
                        print(f"   尝试的查询: {messages[0]['content'][:50]}")
                        # 尝试直接查询数据库验证数据是否存在
                        try:
                            from openjiuwen.core.memory.config.graph import query_expr
                            query_result = self.graph_memory.db_backend.query(
                                collection="episodes",
                                expr=query_expr.filter_user(user_id),
                                limit=10
                            )
                            print(f"   数据库直接查询结果: 找到 {len(query_result)} 个episode（不经过相似度搜索）")
                            if query_result:
                                print(f"   第一个episode内容前100字: {query_result[0].get('content', '')[:100]}")
                        except Exception as e:
                            print(f"   直接查询失败: {e}")
                    
                except Exception as e:
                    print(f"❌ 添加对话历史失败: {e}")
                    import traceback
                    traceback.print_exc()
    
    async def _search_memories(self, query: str, user_id: str) -> Tuple[Dict[str, List[Tuple[float, Any]]], Optional[str]]:
        """从图记忆中检索相关记忆
        
        Returns:
            Tuple[Dict, Optional[str]]: (检索结果字典, 错误信息)
        """
        search_results = {}
        error_msg = None
        
        # 尝试分别检索不同类型，即使某个失败也能返回其他结果
        try:
            print(f"  开始检索记忆: query={query[:50]}..., user_id={user_id}")
            
            # 先尝试检索实体和事件（通常不会有问题）
            try:
                partial_results = self.graph_memory.search(
                    query=query,
                    user_id=user_id,
                    search_strategy="default",
                    entity=True,
                    relation=False,  # 先不检索关系，因为可能有验证错误
                    episode=True,
                )
                search_results.update(partial_results)
            except Exception as e:
                error_msg = f"检索实体/事件失败: {str(e)}"
                print(f"警告: {error_msg}")
            
            # 再尝试检索关系
            try:
                relation_results = self.graph_memory.search(
                    query=query,
                    user_id=user_id,
                    search_strategy="default",
                    entity=False,
                    relation=True,
                    episode=False,
                )
                if "relation" in relation_results:
                    search_results["relation"] = relation_results["relation"]
            except Exception as e:
                relation_error = f"检索关系失败: {str(e)}"
                print(f"警告: {relation_error}")
                if error_msg:
                    error_msg += f"; {relation_error}"
                else:
                    error_msg = relation_error
                # 即使关系检索失败，也继续使用其他结果
            
            # 打印检索结果统计
            entity_count = len(search_results.get("entity", []))
            relation_count = len(search_results.get("relation", []))
            episode_count = len(search_results.get("episode", []))
            print(f"  检索结果: 实体={entity_count}, 关系={relation_count}, 事件={episode_count}")
            if entity_count == 0 and relation_count == 0 and episode_count == 0:
                print(f"  警告: 未检索到任何记忆")
                
            
            return search_results, error_msg
            
        except Exception as e:
            error_msg = f"检索记忆完全失败: {str(e)}"
            print(f"警告: {error_msg}")
            import traceback
            traceback.print_exc()
            return {}, error_msg
    
    async def _generate_answer(
        self,
        question: str,
        retrieved_memories: str,
        background_text: Optional[str] = None,
        conversation_history: Optional[str] = None
    ) -> str:
        """使用 LLM 生成答案"""
        # 构建 prompt
        prompt_parts = []
        
        if retrieved_memories:
            prompt_parts.append("以下是检索到的相关记忆：")
            prompt_parts.append(retrieved_memories)
            prompt_parts.append("")
        
        if background_text and self.use_background:
            prompt_parts.append("背景信息：")
            prompt_parts.append(background_text)
            prompt_parts.append("")
        
        if conversation_history and self.use_conversation:
            prompt_parts.append("对话历史：")
            prompt_parts.append(conversation_history)
            prompt_parts.append("")
        
        prompt_parts.append(f"问题：{question}")
        prompt_parts.append("\n请基于以上信息回答问题。")
        
        prompt = "\n".join(prompt_parts)
        
        try:
            messages = [
                SystemMessage(content="你是一个智能助手，能够基于提供的背景信息、对话历史和检索到的记忆回答问题。请仔细分析所有信息，然后准确回答用户的问题。"),
                HumanMessage(content=prompt)
            ]
            
            model_name = getattr(self.llm_model, '_model_name', "default")
            response = await self.llm_model.ainvoke(
                model_name=model_name,
                messages=messages
            )
            
            return response.content.strip() if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"错误: 生成答案失败: {e}")
            return f"生成答案时出错: {str(e)}"
    
    def _normalize_answer(self, answer: Any) -> str:
        """标准化答案，用于比较"""
        if answer is None:
            return ""
        
        if not isinstance(answer, str):
            answer = str(answer)
        
        return answer.strip()
    
    def _is_answer_correct_simple(self, predicted: Any, expected: Any) -> bool:
        """简单字符串匹配判断答案是否正确"""
        predicted_norm = self._normalize_answer(predicted)
        expected_norm = self._normalize_answer(expected)
        
        if not predicted_norm and not expected_norm:
            return False
        if not predicted_norm or not expected_norm:
            return False
        
        # 精确匹配
        if predicted_norm == expected_norm:
            return True
        
        # 检查预测答案是否包含期望答案
        if expected_norm in predicted_norm:
            return True
        
        # 检查期望答案是否包含预测答案
        if predicted_norm in expected_norm:
            return True
        
        # 尝试数字匹配
        try:
            expected_num = float(expected_norm)
            import re
            numbers = re.findall(r'-?\d+\.?\d*', predicted_norm)
            if numbers:
                for num_str in numbers:
                    if abs(float(num_str) - expected_num) < 0.01:
                        return True
        except (ValueError, TypeError):
            pass
        
        return False
    
    async def _is_answer_correct_llm(
        self,
        predicted: str,
        expected: str,
        question: str,
        question_type: str
    ) -> Tuple[bool, str]:
        """使用 LLM 评估器判断答案是否正确"""
        if question_type in ["时间推理", "time_reasoning"]:
            evaluation_prompt = """你是一个答案评估专家。请评估模型生成的答案是否正确。

问题类型：时间推理（允许时间上的合理偏差）

问题：{question}
期望答案：{expected}
模型答案：{predicted}

请判断模型答案是否正确。对于时间推理问题，如果答案在时间上接近或合理，应该判定为正确。
请以 JSON 格式返回：{{"is_correct": true/false, "reason": "评估理由"}}"""
        else:
            evaluation_prompt = """你是一个答案评估专家。请评估模型生成的答案是否正确。

问题：{question}
期望答案：{expected}
模型答案：{predicted}

请判断模型答案是否正确。答案不需要完全一致，只要语义上正确即可。
请以 JSON 格式返回：{{"is_correct": true/false, "reason": "评估理由"}}"""
        
        prompt = evaluation_prompt.format(
            question=question,
            expected=expected,
            predicted=predicted
        )
        
        try:
            messages = [
                SystemMessage(content="你是一个专业的答案评估专家，能够准确判断答案的正确性。"),
                HumanMessage(content=prompt)
            ]
            
            model_name = getattr(self.evaluator_model, '_model_name', "default")
            response = await self.evaluator_model.ainvoke(
                model_name=model_name,
                messages=messages
            )
            
            # 解析响应
            import re
            content = response.content.strip() if hasattr(response, 'content') else str(response)
            
            # 尝试提取 JSON
            json_match = re.search(r'\{[^{}]*"is_correct"[^{}]*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                is_correct = result.get("is_correct", False)
                reason = result.get("reason", "")
                return bool(is_correct), reason
            
            # 如果无法解析 JSON，尝试从文本中提取
            if "true" in content.lower() or "正确" in content or "correct" in content.lower():
                return True, content
            elif "false" in content.lower() or "错误" in content or "incorrect" in content.lower():
                return False, content
            
            return False, f"无法解析评估结果: {content}"
            
        except Exception as e:
            is_correct = self._is_answer_correct_simple(predicted, expected)
            return is_correct, f"LLM 评估失败，使用简单匹配: {str(e)}"
    
    async def evaluate_item(self, item: LongMemEvalItem) -> EvaluationResult:
        """评估单个测试用例"""
        # 使用 background_id 作为 user_id，这样相同背景的问题会共享记忆
        user_id = f"user_{item.background_id}" if item.background_id else f"user_{item.id}"
        
        try:
            # 添加记忆
            await self._add_memories(item, user_id)
            # 确保所有数据都已刷新
            self.graph_memory.db_backend.refresh()
            await asyncio.sleep(1)  # 等待数据可搜索 
            
            # 检索相关记忆
            search_results, search_error = await self._search_memories(item.question, user_id)
            retrieved_memories_str = self._format_retrieved_memories(search_results)
            
            # 格式化背景和对话历史
            background_text = item.background_text if self.use_background else None
            conversation_history = self._format_conversation_history(item.conversations) if self.use_conversation else None
            
            # 生成答案
            predicted_answer = await self._generate_answer(
                question=item.question,
                retrieved_memories=retrieved_memories_str,
                background_text=background_text,
                conversation_history=conversation_history
            )
            
            # 判断是否正确
            if self.use_llm_evaluator:
                is_correct, reason = await self._is_answer_correct_llm(
                    predicted_answer, item.answer, item.question, item.question_type
                )
            else:
                is_correct = self._is_answer_correct_simple(predicted_answer, item.answer)
                reason = None
            
            # 记录检索到的记忆信息
            retrieved_info = {
                "entity_count": len(search_results.get("entity", [])),
                "relation_count": len(search_results.get("relation", [])),
                "episode_count": len(search_results.get("episode", [])),
                "formatted_content": retrieved_memories_str,  # 格式化的记忆内容
                "search_error": search_error,  # 检索错误信息（如果有）
            }
            
            # 添加详细的记忆内容（用于调试和分析）
            detailed_memories = {}
            
            # 实体详情
            if "entity" in search_results and search_results["entity"]:
                detailed_memories["entities"] = []
                for score, entity in search_results["entity"][:self.search_top_k]:
                    entity_detail = {
                        "name": entity.name if hasattr(entity, 'name') else None,
                        "content": entity.content if hasattr(entity, 'content') else None,
                        "score": score,
                    }
                    if hasattr(entity, 'attributes') and entity.attributes:
                        entity_detail["attributes"] = entity.attributes
                    detailed_memories["entities"].append(entity_detail)
            
            # 关系详情
            if "relation" in search_results and search_results["relation"]:
                detailed_memories["relations"] = []
                for score, relation in search_results["relation"][:self.search_top_k]:
                    relation_detail = {
                        "name": relation.name if hasattr(relation, 'name') else None,
                        "content": relation.content if hasattr(relation, 'content') else None,
                        "score": score,
                    }
                    # 尝试获取 lhs 和 rhs（可能是 UUID 字符串）
                    if hasattr(relation, 'lhs'):
                        relation_detail["lhs"] = relation.lhs if isinstance(relation.lhs, str) else getattr(relation.lhs, 'uuid', str(relation.lhs))
                    if hasattr(relation, 'rhs'):
                        relation_detail["rhs"] = relation.rhs if isinstance(relation.rhs, str) else getattr(relation.rhs, 'uuid', str(relation.rhs))
                    detailed_memories["relations"].append(relation_detail)
            
            # 事件详情
            if "episode" in search_results and search_results["episode"]:
                detailed_memories["episodes"] = []
                for score, episode in search_results["episode"][:self.search_top_k]:
                    episode_detail = {
                        "content": episode.content if hasattr(episode, 'content') else None,
                        "score": score,
                    }
                    detailed_memories["episodes"].append(episode_detail)
            
            retrieved_info["detailed_memories"] = detailed_memories
            
            return EvaluationResult(
                item_id=item.id,
                question=item.question,
                expected_answer=item.answer,
                predicted_answer=predicted_answer,
                question_type=item.question_type,
                is_correct=is_correct,
                evaluation_reason=reason,
                retrieved_memories=retrieved_info
            )
            
        except Exception as e:
            return EvaluationResult(
                item_id=item.id,
                question=item.question,
                expected_answer=item.answer,
                predicted_answer="",
                question_type=item.question_type,
                is_correct=False,
                error=str(e)
            )
    
    async def evaluate_batch(self, items: List[LongMemEvalItem]) -> List[EvaluationResult]:
        """批量评估"""
        results = []
        for i, item in enumerate(items):
            print(f"评估进度: {i+1}/{len(items)} - {item.id}")
            result = await self.evaluate_item(item)
            results.append(result)
        return results
    
    def calculate_metrics(self, results: List[EvaluationResult]) -> EvaluationMetrics:
        """计算评估指标"""
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / total if total > 0 else 0.0
        
        # 按问题类型分组统计
        by_type = {}
        for result in results:
            q_type = result.question_type
            if q_type not in by_type:
                by_type[q_type] = {"total": 0, "correct": 0}
            by_type[q_type]["total"] += 1
            if result.is_correct:
                by_type[q_type]["correct"] += 1
        
        # 计算各类型的准确率
        for q_type in by_type:
            stats = by_type[q_type]
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        
        return EvaluationMetrics(
            total=total,
            correct=correct,
            accuracy=accuracy,
            by_type=by_type
        )
    
    def save_results(self, results: List[EvaluationResult], metrics: EvaluationMetrics, output_dir: str):
        """保存评估结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results_file = os.path.join(output_dir, f"results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
        
        # 保存指标摘要
        metrics_file = os.path.join(output_dir, f"metrics_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            metrics_dict = {
                "total": metrics.total,
                "correct": metrics.correct,
                "accuracy": metrics.accuracy,
                "by_type": metrics.by_type
            }
            json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
        
        # 打印摘要
        print("\n" + "="*60)
        print("评估结果摘要")
        print("="*60)
        print(f"总测试用例数: {metrics.total}")
        print(f"正确答案数: {metrics.correct}")
        print(f"总体准确率: {metrics.accuracy:.2%}")
        print("\n按问题类型统计:")
        for q_type, stats in metrics.by_type.items():
            print(f"  {q_type}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")
        print(f"\n详细结果已保存到: {results_file}")
        print(f"指标摘要已保存到: {metrics_file}")
        print("="*60)


def init_graph_memory(
    storage_path: Optional[str] = None,
    language: str = "cn",
    llm_config: Optional[LLMConfig] = None
) -> GraphMemory:
    """初始化图记忆
    
    Args:
        storage_path: 存储路径（如果为 None，使用临时目录）
        language: 语言设置（cn 或 en）
        llm_config: LLM 配置（如果为 None，从环境变量 JIUWEN_GRAPH_MEM_LLM_* 读取）
    """
    # 配置图记忆数据库
    local_endpoint = os.getenv("LOCAL_ENDPOINT")
    db_uri = f"http://{local_endpoint}"
    print(f"db_uri: {db_uri}")
    
    db_config = GraphConfig(
        uri=db_uri,
        name="evaluation",  # Milvus 数据库名称
        backend="milvus",  # 使用 Milvus 后端
        wipe_at_startup=False,  # 不删除已有数据
        timeout=5.0,  
    )
    
    # 配置 LLM（用于实体提取等）
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
        extraction_strategy=DEFAULT_STRATEGY,
    )
    
    print("✅ 图记忆初始化成功")
    print(f"   存储路径: {db_uri}")
    print(f"   语言: {language}")
    print(f"   LLM 模型: {llm_config.model_name}")
    
    return graph_memory


async def main():
    parser = argparse.ArgumentParser(description="LongMemEval 图记忆评估脚本")
    parser.add_argument("--questions_path", type=str, required=True,
                        help="问题文件路径")
    parser.add_argument("--backgrounds_path", type=str, required=True,
                        help="背景文件路径")
    parser.add_argument("--sessions_paths", type=str, nargs="+", required=True,
                        help="会话文件路径列表")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制测试用例数量（用于快速测试）")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="结果输出目录 (默认: ./eval_results)")
    parser.add_argument("--use_background", action="store_true", default=True,
                        help="是否使用背景文本 (默认: True)")
    parser.add_argument("--use_conversation", action="store_true", default=True,
                        help="是否使用对话历史 (默认: True)")
    parser.add_argument("--use_llm_evaluator", action="store_true", default=False,
                        help="是否使用 LLM 评估器（符合 LongMemEval 官方标准，默认: False）")
    parser.add_argument("--search_top_k", type=int, default=5,
                        help="检索记忆时的 top_k 数量 (默认: 5)")
    parser.add_argument("--storage_path", type=str, default=None,
                        help="图记忆存储路径（默认: 临时目录）")
    parser.add_argument("--language", type=str, choices=["cn", "en"], default="cn",
                        help="图记忆语言 (默认: cn)")
    
    # 模型配置参数
    parser.add_argument("--api_base", type=str, default=None,
                        help="LLM API base URL (或使用环境变量 API_BASE)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="LLM API key (或使用环境变量 API_KEY)")
    parser.add_argument("--model_name", type=str, default=None,
                        help="模型名称 (或使用环境变量 MODEL_NAME)")
    parser.add_argument("--model_provider", type=str, default=None,
                        help="模型提供商 (或使用环境变量 MODEL_PROVIDER)")
    
    # 图记忆 LLM 配置（用于实体提取等）
    parser.add_argument("--graph_llm_api_base", type=str, default=None,
                        help="图记忆 LLM API base URL (或使用环境变量 JIUWEN_GRAPH_MEM_LLM_URL)")
    parser.add_argument("--graph_llm_api_key", type=str, default=None,
                        help="图记忆 LLM API key (或使用环境变量 JIUWEN_GRAPH_MEM_LLM_KEY)")
    parser.add_argument("--graph_llm_model_name", type=str, default=None,
                        help="图记忆 LLM 模型名称 (或使用环境变量 JIUWEN_GRAPH_MEM_LLM_MODEL)")

    # LLM 评估器配置
    parser.add_argument("--evaluator_api_base", type=str, default=None,
                        help="评估器 LLM API base URL (或使用环境变量 EVALUATOR_API_BASE)")
    parser.add_argument("--evaluator_api_key", type=str, default=None,
                        help="评估器 LLM API key (或使用环境变量 EVALUATOR_API_KEY)")
    parser.add_argument("--evaluator_model_name", type=str, default=None,
                        help="评估器模型名称 (或使用环境变量 EVALUATOR_MODEL_NAME)")
    parser.add_argument("--evaluator_model_provider", type=str, default=None,
                        help="评估器模型提供商 (或使用环境变量 EVALUATOR_MODEL_PROVIDER)")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ.setdefault("LLM_SSL_VERIFY", "false")
    
    # 获取主模型配置 用于回答问题
    api_base = args.api_base or os.getenv("API_BASE") or os.getenv("JIUWEN_GRAPH_MEM_LLM_URL")
    api_key = args.api_key or os.getenv("API_KEY") or os.getenv("JIUWEN_GRAPH_MEM_LLM_KEY")
    model_name = args.model_name or os.getenv("MODEL_NAME") or os.getenv("JIUWEN_GRAPH_MEM_LLM_MODEL")
    model_provider = args.model_provider or os.getenv("MODEL_PROVIDER")
    
    if not all([api_base, api_key, model_name, model_provider]):
        print("错误: 请提供主模型配置参数或设置环境变量")
        print("需要以下之一：")
        print("  1. API_BASE, API_KEY, MODEL_NAME, MODEL_PROVIDER")
        print("  2. JIUWEN_GRAPH_MEM_LLM_URL, JIUWEN_GRAPH_MEM_LLM_KEY, JIUWEN_GRAPH_MEM_LLM_MODEL, MODEL_PROVIDER")
        return
    
    # 获取图记忆 LLM 配置 用于实体提取等
    graph_llm_config = None
    if args.graph_llm_api_base or args.graph_llm_api_key or args.graph_llm_model_name:
        # 如果提供了命令行参数，手动创建配置
        graph_llm_api_base = args.graph_llm_api_base or os.getenv("JIUWEN_GRAPH_MEM_LLM_URL") or api_base
        graph_llm_api_key = args.graph_llm_api_key or os.getenv("JIUWEN_GRAPH_MEM_LLM_KEY") or api_key
        graph_llm_model_name = args.graph_llm_model_name or os.getenv("JIUWEN_GRAPH_MEM_LLM_MODEL") or model_name
        
        # 尝试从环境变量读取 CONFIG
        graph_llm_config_str = os.getenv("JIUWEN_GRAPH_MEM_LLM_CONFIG", "{}")
        try:
            graph_llm_config_dict = json.loads(graph_llm_config_str)
        except json.JSONDecodeError:
            graph_llm_config_dict = {}
        
        graph_llm_config = LLMConfig(
            api_key=graph_llm_api_key,
            api_base=graph_llm_api_base,
            model_name=graph_llm_model_name,
            **graph_llm_config_dict
        )
        print("✅ 使用命令行参数或环境变量创建图记忆 LLM 配置")
    else:
        # 尝试从环境变量自动读取
        try:
            graph_llm_config = LLMConfig.default_config()
            print("✅ 从环境变量 JIUWEN_GRAPH_MEM_LLM_* 自动读取图记忆 LLM 配置")
        except (ValueError, KeyError):
            # 如果无法读取，使用主模型配置
            graph_llm_config = LLMConfig(
                api_key=api_key,
                api_base=api_base,
                model_name=model_name,
            )
            print("⚠️  无法从环境变量读取图记忆 LLM 配置，使用主模型配置")
    
    # 初始化图记忆
    print("初始化图记忆...")
    graph_memory = init_graph_memory(
        storage_path=args.storage_path,
        language=args.language,
        llm_config=graph_llm_config
    )
    
    # 创建主 LLM 模型（用于回答问题）
    print(f"创建主 LLM 模型: {model_name}...")
    main_llm = ModelFactory().get_model(
        model_provider=model_provider,
        api_key=api_key,
        api_base=api_base
    )
    # 保存 model_name 以便后续使用
    main_llm._model_name = model_name
    
    # 创建评估器模型（如果使用 LLM 评估器）
    evaluator_model = None
    if args.use_llm_evaluator:
        evaluator_api_base = args.evaluator_api_base or os.getenv("EVALUATOR_API_BASE") or api_base
        evaluator_api_key = args.evaluator_api_key or os.getenv("EVALUATOR_API_KEY") or api_key
        evaluator_model_name = args.evaluator_model_name or os.getenv("EVALUATOR_MODEL_NAME") or model_name
        evaluator_model_provider = args.evaluator_model_provider or os.getenv("EVALUATOR_MODEL_PROVIDER") or model_provider
        
        evaluator_llm = ModelFactory().get_model(
            model_provider=evaluator_model_provider,
            api_key=evaluator_api_key,
            api_base=evaluator_api_base
        )
        evaluator_llm._model_name = evaluator_model_name
        evaluator_model = evaluator_llm
        print(f"使用 LLM 评估器: {evaluator_model_name} (符合 LongMemEval 官方标准)")
    else:
        print("使用简单字符串匹配评估（快速测试模式）")
        print("提示: 使用 --use_llm_evaluator 启用 LLM 评估器，符合 LongMemEval 官方标准")
    
    # 加载数据
    print("加载 LongMemEval 数据集...")
    loader = LongMemEvalLoader(
        questions_path=args.questions_path,
        backgrounds_path=args.backgrounds_path,
        sessions_paths=args.sessions_paths
    )
    items = loader.load(limit=args.limit)
    print(f"加载了 {len(items)} 个测试用例")
    
    # 创建评估器
    evaluator = GraphMemoryEvaluator(
        graph_memory=graph_memory,
        llm_model=main_llm,
        use_background=args.use_background,
        use_conversation=args.use_conversation,
        use_llm_evaluator=args.use_llm_evaluator,
        evaluator_model=evaluator_model,
        search_top_k=args.search_top_k
    )
    
    # 执行评估
    print("\n开始评估...")
    results = await evaluator.evaluate_batch(items)
    
    # 计算指标
    metrics = evaluator.calculate_metrics(results)
    
    # 保存结果
    evaluator.save_results(results, metrics, args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
