# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import asyncio
from typing import Iterable, List, Optional, Tuple, Union

from llama_index.core.schema import TextNode

from openjiuwen.core.common.logging import logger
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.integrations.retriever.config.configuration import CONFIG
from openjiuwen.integrations.retriever.retrieval.llms.client import (
    get_llm_client,
    get_model_name,
)
from openjiuwen.integrations.retriever.retrieval.search.agents.prompts.read import (
    get_read_prompt,
    postproc_read,
)
from openjiuwen.integrations.retriever.retrieval.search.agents.prompts.reason import (
    get_reason_prompt,
    postproc_reason,
)
from openjiuwen.integrations.retriever.retrieval.search.agents.prompts.rewrite import (
    get_rewrite_prompt,
    postproc_rewrite,
)
from openjiuwen.integrations.retriever.retrieval.search.agents.triple_memory import (
    TripleMemory,
)
from openjiuwen.integrations.retriever.retrieval.search.fusion import GraphRetriever
from openjiuwen.integrations.retriever.retrieval.search.milvus import rrf_nodes
from openjiuwen.integrations.retriever.retrieval.search.retrieval_models import (
    BaseRetriever as _BaseRetriever,
)
from openjiuwen.integrations.retriever.retrieval.search.retrieval_models import (
    Dataset,
    Document,
    RetrievalResult,
    TextChunk,
)
from openjiuwen.integrations.retriever.retrieval.utils import deduplicate


class SearchAgent(_BaseRetriever):
    def __init__(
        self,
        retriever: GraphRetriever,
        retriever_config: dict,
        max_iter: int = 4,
        topk: int = 15,
        use_sync: bool = True,
        use_agent: bool = False,
        mode: str = "hybrid",
        config_obj=None,
        llm_client: BaseModelClient | None = None,
        verbose: bool = False,
    ):
        cfg = config_obj or CONFIG
        if llm_client is None:
            if cfg is None:
                raise ValueError(
                    "llm_client is required for SearchAgent (no default config)"
                )
            llm_client = get_llm_client(config=cfg)
        self.llm: BaseModelClient = llm_client
        self._config = cfg
        self.retriever = retriever
        self._retriever_config = retriever_config
        self.max_iter = max_iter
        self.topk = topk
        self.use_sync = use_sync
        self.use_agent = use_agent
        self._graph_exp_config = retriever_config.get("graph_expansion_config", {})
        self._retrieval_mode = mode
        self._verbose = verbose

    def __call__(
        self, query: Union[str, Iterable[str]], batch_size: int = 8, *args, **kwargs
    ):
        """Not yet implemented"""
        pass

    def _llm_call(self, prompt: str) -> str:
        """
        Wrapper function for calling the designated LLM.

        Args:
            prompt (str): input prompt

        Returns:
            str: completion output coming from the LLM of choice
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(
            model_name=get_model_name(self._config), messages=messages, temperature=0.0
        )
        return response.content

    def _log_verbose(
        self,
        tag: str,
        prompt: str | None = None,
        completion: str | None = None,
        extra: str | None = None,
    ):
        if not self._verbose:
            return
        parts = [f"[{tag}]"]
        if extra:
            parts.append(extra)
        if prompt is not None:
            parts.append(f"prompt={prompt!r}")
        if completion is not None:
            parts.append(f"completion={completion!r}")
        logger.info(" ".join(parts))

    async def read(
        self,
        query: str,
        passages: List[TextNode],
        triple_memory: Union[TripleMemory, None] = None,
    ) -> List[Tuple[str, ...]]:
        """
        Function responsible for running the `read` step within GeAR.

        Args:
            query (str):
            passages (List[TextNode]):
            triple_memory (Union[TripleMemory, None], optional): Defaults to None.

        Returns:
            List[Tuple[str, ...]]: list of `tuple`-ed triples
        """

        triples = triple_memory.memory if triple_memory is not None else None
        prompt = get_read_prompt(query=query, passages=passages, triples=triples)
        completion = self._llm_call(prompt)
        self._log_verbose("READ", prompt=prompt, completion=completion)
        logger.debug("READ\nprompt=%r\ncompletion=%r", prompt, completion)
        return postproc_read(completion=completion)

    async def retrieve(self, query: str) -> List[TextNode]:
        return await self.retriever.async_search(query=query, mode=self._retrieval_mode)

    async def reason(
        self, query: str, triple_memory: Union[TripleMemory, None]
    ) -> Tuple[bool, str]:
        """
        Function responsible for running the `reason` step within GeAR.

        Args:
            query (str):
            triple_memory (Union[TripleMemory, None], optional):

        Returns:
            Tuple[bool, str]:
        """

        prompt = get_reason_prompt(query=query, triples=triple_memory.memory)
        completion = self._llm_call(prompt)
        self._log_verbose("REASON", prompt=prompt, completion=completion)
        logger.debug("REASON\nprompt=%r\ncompletion=%r", prompt, completion)
        return postproc_reason(completion=completion)

    async def rewrite(
        self, query: str, triple_memory: Union[TripleMemory, None], reason: str
    ) -> str:
        """
        Function responsible for running the query `rewrite` step within GeAR.

        Args:
            query (str):
            triple_memory (Union[TripleMemory, None], optional):
            reason (str):

        Returns:
            str: the query to be used at the subsequent iteration
        """

        triples = triple_memory.memory if len(triple_memory) > 0 else None
        prompt = get_rewrite_prompt(query=query, triples=triples, reason=reason)
        completion = self._llm_call(prompt)
        self._log_verbose("REWRITE", prompt=prompt, completion=completion)
        logger.debug("QUERY RE-WRITE\nprompt=%r\ncompletion=%r", prompt, completion)
        return postproc_rewrite(completion=completion)

    async def link_passages(self, triple_memory: TripleMemory) -> List[List[TextNode]]:
        triples = triple_memory.memory
        if not triples:
            return []

        return await asyncio.gather(
            *[
                self.retriever.chunk_retriever.async_search(
                    query=" ".join(tmp_triple),
                    mode="hybrid",
                    topk=5,
                )
                for tmp_triple in triples
            ]
        )

    async def link_triples(self, triples: List[str]) -> List[List[TextNode]]:

        nodes = await asyncio.gather(
            *[
                self.retriever.triple_retriever.async_search(
                    query=" ".join(tmp_triple), mode="hybrid", topk=1
                )
                for tmp_triple in triples
            ]
        )
        return deduplicate([x[0] for x in nodes], key=lambda node: node.node_id)

    async def search(self, query: str) -> List[TextNode]:
        """
        Function responsible for the mainline search functionality of `SearchAgent`.

        Args:
            query (str): input natural language query

        Returns:
            List[TextNode]: list of passages relevant to the original `query`
        """

        queries = [query]
        memory = TripleMemory()
        all_passages = []
        for turn in range(1, self.max_iter + 1):
            running_query = queries[-1]
            self._log_verbose("TURN", extra=f"turn={turn} query={running_query!r}")
            passages = await self.retrieve(running_query)
            if self._verbose:
                logger.info("[TURN %d] retrieved %d passages", turn, len(passages))

            running_triples = None
            if self.use_sync:
                proximal_triples = await self.read(
                    query=running_query, passages=passages, triple_memory=None
                )
                logger.debug(
                    "After the first-read in turn=%r we get proximal_triples=%r",
                    turn,
                    proximal_triples,
                )
                if len(proximal_triples) == 0:
                    logger.warning(
                        "self.use_sync=%r but no proximal triples are returned. Falling back to NaiveGE ...",
                        self.use_sync,
                    )
                else:
                    running_triples = await self.link_triples(proximal_triples)

            passages = await self.retriever.graph_expansion(
                query=running_query,
                triples=running_triples,
                chunks=passages,
                **self._graph_exp_config,
            )
            if not self.use_agent:
                return passages

            triples = await self.read(query, passages=passages, triple_memory=memory)
            if self._verbose:
                logger.info("[TURN %d] triples_after_read=%r", turn, triples)
            logger.debug("第二次抽取后，turn=%r 得到 triples=%r", turn, triples)
            memory.batch_extend_memory(triples)
            logger.debug(
                "扩展记忆后，turn=%r memory=%r",
                turn,
                memory.memory,
            )
            all_passages.append(passages)
            logger.info(
                "Agent 检索回合=%d：本轮候选chunk=%d，累计候选组数=%d",
                turn,
                len(passages),
                len(all_passages),
            )

            answer_or_reason = ""

            if len(memory) > 0:
                is_answerable, answer_or_reason = await self.reason(
                    query, triple_memory=memory
                )
                if self._verbose:
                    logger.info(
                        "[TURN %d] reason answerable=%r text=%r",
                        turn,
                        is_answerable,
                        answer_or_reason,
                    )

                if is_answerable:
                    logger.info("Agent 在 turn=%r 成功回答 query=%r", turn, query)
                    break
            if turn >= self.max_iter:
                break

            next_query = await self.rewrite(
                query, triple_memory=memory, reason=answer_or_reason
            )
            if self._verbose:
                logger.info("[TURN %d] rewritten query=%r", turn, next_query)
            if not next_query:
                break
            queries.append(next_query)
        ret = await self.link_passages(memory)
        # 如果 agent 没有返回引用节点，至少回退到已检索到的 chunk 结果，避免空结果
        combined = rrf_nodes(ret + all_passages)[: self.topk]
        logger.info(
            "Agent 结束：可引用的chunk=%d，SearchAgent回合数=%d，rrf融合后chunk数=%d",
            sum(len(g) for g in ret),
            len(all_passages),
            len(combined),
        )
        if combined:
            return combined

        if all_passages:
            logger.warning(
                "Agent 未返回引用节点，回退到检索到的 chunk 结果（组数=%d，每组示例大小=%s）",
                len(all_passages),
                [len(g) for g in all_passages[:3]],
            )
            # rrf_nodes 的参数是 k（融合超参），这里先融合再截断 topk
            fallback = rrf_nodes(all_passages)[: self.topk]
            if fallback:
                return fallback
            # 保险兜底：简单平铺 chunk
            flat = [n for group in all_passages for n in group][: self.topk]
            return flat

        logger.warning("Agent 未检索到任何 chunk，返回空结果")
        return []

    def list_datasets(self, *args, **kwargs):
        return self.retriever.list_datasets(*args, **kwargs)

    def list_documents(self, *args, **kwargs):
        return self.retriever.list_documents(*args, **kwargs)

    def search_relevant_documents(
        self, question: str, datasets: Optional[List[Dataset]] = None, *args, **kwargs
    ) -> RetrievalResult:
        if datasets is None:
            datasets = []
        dataset_set = {(dataset.title, dataset.uri) for dataset in datasets}
        self_dataset_set = {
            (dataset.title, dataset.uri) for dataset in self.list_datasets()
        }
        if dataset_set and dataset_set != self_dataset_set:
            return []

        results = asyncio.get_event_loop().run_until_complete(
            self.search(query=question)
        )
        result = RetrievalResult(
            query=question,
            datasets=self.list_datasets(),
            documents=[
                Document(
                    document_id=doc.id_,
                    title=doc.metadata["title"],
                    url=(
                        None
                        if CONFIG.get("hide_local_urls", True)
                        else f"{self.retriever.chunk_retriever.dataset.uri}/_doc/{doc.id_}"
                    ),
                    chunks=[TextChunk(content=doc.text, similarity_score=1.0)],
                )
                for doc in results
            ],
        )
        return result
