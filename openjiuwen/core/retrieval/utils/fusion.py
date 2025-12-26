# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
融合工具函数

包含 RRF (Reciprocal Rank Fusion) 等融合算法。
"""
from typing import List, Dict, Any
from collections import defaultdict

from openjiuwen.core.retrieval.common.retrieval_result import RetrievalResult, SearchResult


def rrf_fusion(
    results_list: List[List[RetrievalResult]],
    k: int = 60,
) -> List[RetrievalResult | SearchResult]:
    """
    Reciprocal Rank Fusion (RRF) 融合多个检索结果
    
    Args:
        results_list: 多个检索结果列表
        k: RRF 参数，默认 60
        
    Returns:
        融合后的检索结果列表
    """
    # 使用字典存储每个结果的分数
    score_dict: Dict[str, float] = defaultdict(float)
    result_dict: Dict[str, RetrievalResult] = {}
    
    # 对每个结果列表进行融合
    for results in results_list:
        for rank, result in enumerate(results, start=1):
            # 使用文本作为唯一标识
            key = result.text
            # RRF 分数计算
            score_dict[key] += 1.0 / (k + rank)
            # 保存结果对象（保留第一个出现的元数据）
            if key not in result_dict:
                result_dict[key] = result
    
    # 按分数排序
    sorted_items = sorted(
        score_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # 构建融合后的结果列表
    fused_results = []
    for key, score in sorted_items:
        result = result_dict[key]
        # 更新分数为融合后的分数
        result.score = score
        fused_results.append(result)
    
    return fused_results
