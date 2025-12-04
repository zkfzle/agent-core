# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from openjiuwen.core.common.logging import logger
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.integrations.retriever.config.configuration import CONFIG as default_config
from openjiuwen.integrations.retriever.doc_process.components.extraction.extract_triples import extract_triples
from openjiuwen.integrations.retriever.doc_process.components.indexing.index import index
from openjiuwen.integrations.retriever.doc_process.components.indexing.index_triples import index_triples
from openjiuwen.integrations.retriever.retrieval.embed_models.base import EmbedModel


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

    def verify_indices(self) -> None:
        """验证索引构建结果"""
        logger.info("\n验证构建结果...")

        try:
            import requests

            indices_info = [(self.config.chunk_es_index, "文本索引"), (self.config.triple_es_index, "三元组索引")]

            for index_name, index_desc in indices_info:
                response = requests.get(f"{self.config.es_url}/{index_name}/_count")
                if response.status_code == 200:
                    count = response.json()["count"]
                    logger.info("{index_desc}文档数: %r", count)
                else:
                    logger.error("{index_desc}检查失败: %r", response.status_code)

        except ImportError:
            logger.warning("缺少requests库，无法验证结果")
        except Exception as e:
            logger.warning("验证结果时出错: %r", e)


class GraphRAGIndexBuilder:
    """索引构建器"""

    def __init__(self, config_obj=None, config_file: Optional[str] = None, file: Optional[Dict[str, str]] = None):
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
        self, skip_text_index: bool = False, skip_triple_extraction: bool = False, skip_triple_index: bool = False
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
        if not skip_text_index:
            await build_text_index()
        if not skip_triple_extraction:
            chunk2triples = await build_triple_extraction()
        if not skip_triple_index:
            await build_triple_index(chunk2triples=chunk2triples)

        # 验证结果
        self.result_verifier.verify_indices()

        logger.info("索引构建完成！")
        logger.info("=" * 60)


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
            config_obj=config.config_obj or default_config, config_file=config.config_file, file=config.file
        )

        await builder.build(
            skip_text_index=config.skip_text_index,
            skip_triple_extraction=config.skip_triple_extraction,
            skip_triple_index=config.skip_triple_index,
        )

    except KeyboardInterrupt:
        logger.info("\n 构建被用户中断")
        raise
    except Exception as e:
        logger.exception("构建过程中发生未预期的错误", e)
        raise
