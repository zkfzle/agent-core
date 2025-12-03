# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import time
from dataclasses import dataclass, field
from pathlib import Path
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


class EnvironmentChecker:
    """环境检查器"""

    def __init__(self, config_obj):
        self.config = config_obj

    def check_elasticsearch(self) -> bool:
        """检查Elasticsearch连接"""
        try:
            import requests

            response = requests.get(f"{self.config.es_url}/_cluster/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Elasticsearch连接正常")
                return True
            else:
                logger.error("❌ Elasticsearch连接失败: %r", response.status_code)
                return False
        except ImportError:
            logger.error("❌ 缺少requests库，请安装: pip install requests")
            raise
        except Exception as e:
            logger.error("❌ Elasticsearch连接失败: %r", e)
            raise

    def check_input_file(self) -> bool:
        """检查输入文件"""
        input_file = self.config.get_full_data_path(self.config.input_data_file)
        if not input_file.exists():
            logger.error("❌ 输入文件不存在: %r", input_file)
            return False
        logger.info("✅ 输入文件存在: %r", input_file)
        return True

    def check_data_directory(self, project_root: Path) -> bool:
        """检查数据目录"""
        data_dir = project_root / self.config.data_dir
        if not data_dir.exists():
            logger.error("❌ 数据目录不存在: %r", data_dir)
            return False
        logger.info("✅ 数据目录存在: %r", data_dir)
        return True

    def run_all_checks(self, project_root: Path) -> bool:
        """运行所有环境检查"""
        logger.info("🔍 检查环境配置...")

        checks = [
            self.check_elasticsearch,
            self.check_input_file,
            lambda: self.check_data_directory(project_root),
        ]

        return all(check() for check in checks)


async def run_function(func: callable, description: str, parameters: Optional[dict] = None) -> bool:
    """运行函数并显示结果"""
    logger.info("\n🔄 %r", description)
    logger.info("   函数: %r", func.__name__)

    if parameters:
        logger.info("   参数: %r", parameters)

    start_time = time.time()
    try:
        # 调用函数，如果有参数则传递参数
        result = await func(**parameters)
        end_time = time.time()
        log_msg = f"✅ {description} 成功 (耗时: {end_time - start_time:.2f}秒)"
        logger.info(log_msg)
    except Exception as e:
        logger.exception("❌ %r 执行异常", description)
        raise

    return result


class ResultVerifier:
    """结果验证器"""

    def __init__(self, config_obj):
        self.config = config_obj

    def verify_indices(self) -> None:
        """验证索引构建结果"""
        logger.info("\n🔍 验证构建结果...")

        try:
            import requests

            indices_info = [(self.config.chunk_es_index, "文本索引"), (self.config.triple_es_index, "三元组索引")]

            for index_name, index_desc in indices_info:
                response = requests.get(f"{self.config.es_url}/{index_name}/_count")
                if response.status_code == 200:
                    count = response.json()["count"]
                    logger.info("✅ {index_desc}文档数: %r", count)
                else:
                    logger.error("❌ {index_desc}检查失败: %r", response.status_code)

        except ImportError:
            logger.warning("⚠️ 缺少requests库，无法验证结果")
        except Exception as e:
            logger.warning("⚠️ 验证结果时出错: %r", e)


class GraphRAGIndexBuilder:
    """索引构建器"""

    def __init__(self, config_obj=None, config_file: Optional[str] = None, file: Optional[Dict[str, str]] = None):
        self.config_file = config_file
        self.config = config_obj or default_config
        if self.config is None:
            raise ValueError("config_obj (GraphRAGConfig) is required")
        self.file = file

        # 自动找到项目根目录
        self.project_root = self._find_project_root()

        # 初始化组件
        self.env_checker = EnvironmentChecker(self.config)
        self.result_verifier = ResultVerifier(self.config)

    @staticmethod
    def _find_project_root() -> Path:
        """自动找到项目根目录（包含 pyproject.toml 的目录）"""
        current_dir = Path(__file__).parent
        while current_dir != current_dir.parent:
            if (current_dir / "pyproject.toml").exists():
                return current_dir
            current_dir = current_dir.parent
        raise RuntimeError("找不到项目根目录（包含 pyproject.toml 的目录）")

    def print_header(self) -> None:
        """打印脚本头部信息"""
        logger.info("🚀 索引构建脚本")
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

        # 检查环境
        # if not self.env_checker.run_all_checks(self.project_root):
        #     logger.error("❌ 环境检查失败，退出构建")
        #     return False

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

        logger.info("\n🎉 索引构建完成！")
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
        logger.info("\n⚠️ 构建被用户中断")
        raise
    except Exception as e:
        logger.exception("❌ 构建过程中发生未预期的错误")
        raise
