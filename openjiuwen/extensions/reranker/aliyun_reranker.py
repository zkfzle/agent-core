# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
External Reranker Model Implementation

Reranker client implementation for Aliyun / Bailian
"""

from openjiuwen.core.retrieval.reranker.standard_reranker import StandardReranker


class AliyunReranker(StandardReranker):
    """
    Aliyun reranker client, supports text-rerank API of Alibaba Cloud / Aliyun / Bailian
    """

    end_point = "/rerank/text-rerank/text-rerank"

    def _request_params(self, **kwargs: dict) -> dict:
        documents = kwargs["documents"]
        return {
            "model": self.model_name,
            "input": dict(query=kwargs["query"], documents=documents),
            "parameters": dict(return_documents=False, top_n=kwargs.get("top_n", len(documents))),
        }
