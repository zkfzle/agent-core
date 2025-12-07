# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from openjiuwen.core.common.logging import logger
from openjiuwen.integrations.retriever.config.configuration import (
    CONFIG as default_config,
)
from openjiuwen.integrations.retriever.doc_process.components.extraction.extract_triples import (
    extract_triples,
)
from openjiuwen.integrations.retriever.doc_process.components.indexing.index import (
    index,
)
from openjiuwen.integrations.retriever.doc_process.components.indexing.index_triples import (
    index_triples,
)
from openjiuwen.integrations.retriever.retrieval.embed_models.base import EmbedModel
from openjiuwen.integrations.retriever.retrieval.llms.client import BaseModelClient
from openjiuwen.integrations.retriever.retrieval.utils.milvus_client import (
    milvus_manager,
)


@dataclass
class GRAGConfig:
    file: Optional[Dict[str, str]] = field(default=None)
    skip_text_index: bool = field(default=False)
    skip_triple_extraction: bool = field(default=False)
    skip_triple_index: bool = field(default=False)
    config_file: Optional[str] = field(default=None)
    config_obj: Optional[Any] = field(default=None)
    embed_model: Optional[EmbedModel] = field(default=None)
    llm_client: Optional[BaseModelClient] = field(default=None)


class ResultVerifier:
    """结果验证器"""

    def __init__(self, config_obj):
        self.config = config_obj

    async def verify_indices(self) -> None:
        """异步验证索引构建结果，避免阻塞事件循环"""
        logger.info("验证构建结果...")

        def _run():
            try:
                client = milvus_manager.get_client(
                    uri=self.config.milvus_uri,
                    token=self.config.milvus_token,
                )

                indices_info = [
                    (self.config.chunk_es_index, "文本索引"),
                    (self.config.triple_es_index, "三元组索引"),
                ]

                for collection_name, index_desc in indices_info:
                    if client.has_collection(collection_name):
                        stats = client.get_collection_stats(collection_name)
                        count = stats.get("row_count", 0)
                        label = "chunk 数量" if index_desc == "文本索引" else "三元组数量"
                        logger.info(f"{index_desc} ({collection_name}) {label}: {count}")
                    else:
                        logger.warning(f"{index_desc} ({collection_name}) 不存在")

            except Exception as e:
                logger.warning("验证结果时出错: %r", e)
            
            finally:
                if client is not None:
                    milvus_manager.release()

        await asyncio.to_thread(_run)


class GraphRAGIndexBuilder:
    """索引构建器"""

    def __init__(
        self,
        config_obj=None,
        config_file: Optional[str] = None,
        file: Optional[Dict[str, str]] = None,
    ):
        self.config_file = config_file
        self.config = config_obj or default_config
        if self.config is None:
            raise ValueError("config_obj (GraphRAGConfig) is required")
        self.file = file
        # 初始化组件
        self.result_verifier = ResultVerifier(self.config)

    def print_header(self) -> None:
        """打印脚本头部信息"""
        logger.info("索引构建脚本")
        logger.info("=" * 60)
        self.config.print_config()

    async def build(
        self,
        skip_text_index: bool = False,
        skip_triple_extraction: bool = False,
        skip_triple_index: bool = False,
    ) -> bool:
        """执行完整的索引构建流程"""
        self.print_header()

        # 如果配置关闭图索引，则自动跳过三元组相关步骤
        if not getattr(self.config, "use_graph_index", True):
            skip_triple_extraction = True
            skip_triple_index = True

        async def build_text_index():
            await index(
                from_file=self.file,
                config_obj=self.config,
                embed_model=getattr(self.config, "embed_model_instance", None),
            )

        async def build_triple_extraction():
            return await extract_triples(
                file_id=self.file["id"],
                config_obj=self.config,
                llm_client=getattr(self.config, "llm_client_instance", None),
            )

        async def build_triple_index(chunk2triples: dict):
            await index_triples(
                chunk2triples=chunk2triples,
                file_id=self.file["id"],
                config_obj=self.config,
                embed_model=getattr(self.config, "embed_model_instance", None),
            )

        # 执行构建阶段
        chunk2triples = None
        metrics = {
            "text_index_s": None,
            "triple_extract_s": None,
            "triple_index_s": None,
        }
        t_total = time.perf_counter()
        if not skip_text_index:
            t = time.perf_counter()
            logger.info("[步骤 1/3] 开始构建文本索引")
            await build_text_index()
            metrics["text_index_s"] = time.perf_counter() - t
            logger.info("[步骤 1/3] 文本索引完成，用时 %.2fs", metrics["text_index_s"])
        if not skip_triple_extraction:
            t = time.perf_counter()
            logger.info("[步骤 2/3] 开始抽取三元组")
            chunk2triples = await build_triple_extraction()
            metrics["triple_extract_s"] = time.perf_counter() - t
            logger.info(
                "[步骤 2/3] 三元组抽取完成，用时 %.2fs", metrics["triple_extract_s"]
            )
        if not skip_triple_index:
            t = time.perf_counter()
            logger.info("[步骤 3/3] 开始写入三元组索引")
            await build_triple_index(chunk2triples=chunk2triples)
            metrics["triple_index_s"] = time.perf_counter() - t
            logger.info(
                "[步骤 3/3] 三元组索引完成，用时 %.2fs", metrics["triple_index_s"]
            )

        # 验证结果，超时则跳过以免卡住
        verify_timeout = getattr(self.config, "verify_timeout", 3)  # 秒
        try:
            await asyncio.wait_for(self.result_verifier.verify_indices(), timeout=verify_timeout)
        except asyncio.TimeoutError:
            logger.warning("验证构建结果超时(>%ss)，跳过 verify_indices", verify_timeout)

        metrics["total_s"] = time.perf_counter() - t_total
        logger.info("索引构建完成！")
        logger.info("=" * 60)
        return metrics


async def build_grag_index(config: GRAGConfig) -> bool:
    """主函数

    Returns:
        bool: True if build was successful, False otherwise
    """
    try:
        # 如果调用方传入 config/实例，则覆盖默认
        if config.config_obj is not None:
            import openjiuwen.integrations.retriever.config.configuration as cfg_mod

            cfg_mod.CONFIG = config.config_obj
            # 注入实例以便后续使用
            if config.embed_model is not None:
                setattr(config.config_obj, "embed_model_instance", config.embed_model)
            if config.llm_client is not None:
                setattr(config.config_obj, "llm_client_instance", config.llm_client)
        if config.config_obj is None and default_config is None:
            raise ValueError("config_obj is required (GraphRAGConfig)")

        builder = GraphRAGIndexBuilder(
            config_obj=config.config_obj or default_config,
            config_file=config.config_file,
            file=config.file,
        )

        return await builder.build(
            skip_text_index=config.skip_text_index,
            skip_triple_extraction=config.skip_triple_extraction,
            skip_triple_index=config.skip_triple_index,
        )

    except KeyboardInterrupt:
        logger.info("\n 构建被用户中断")
        raise
    except Exception as e:
        logger.exception("构建过程中发生未预期的错误: %s", e)
        raise
