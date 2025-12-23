# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import asyncio
import math
from threading import Thread
from typing import Dict, Iterable, List, Literal, Optional, Union

import httpx
import numpy as np

from openjiuwen.core.memory.generation.graph.prompts import TemplateManager

from .api_requests import async_request_with_retry, sync_request_with_retry
from .llm_client import GraphLLMClient
from .prompts import SUPPORTED_LANGUAGES as reranker_languages


class BaseReranker(GraphLLMClient):
    """Document relevance re-ranking"""

    yes_no_ids: tuple[int, int] = ()
    yes_no_text: tuple[str, str] = ()
    preset: Literal["default", "vllm", "aliyun"] = "default"

    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_base: str,
        temperature: float = 0.6,
        top_p: float = 1.0,
        timeout: float = 60.0,
        structured_output: bool = True,
        preset: Literal["default", "vllm", "aliyun"] = "default",
        **kwargs,
    ):
        """（仅在使用思考模型时使用）
        preset解释：vLLM/SGLang使用\"vllm\"选项传入chat_template_kwargs，阿里云/Siliconflow使用\"aliyun\"选项传入extra_body
        """

        super().__init__(model_name, api_key, api_base, temperature, top_p, timeout, structured_output)
        self.extra_args = kwargs
        self.preset = preset if preset in ["default", "vllm", "aliyun"] else "default"

    def rerank(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None,
        sync_client: Optional[httpx.Client] = None,
        async_client: Optional[httpx.AsyncClient] = None,
        max_retry: int = 30,
        retry_wait: float = 0.1,
        language: str = "en",
        use_mean_score_as_backoff: bool = True,
        **kwargs,
    ) -> Dict[str, float]:
        """Perform cross-encoder reranking on documents

        Args:
            query (str): query string.
            documents (List[str]): list of documents to rerank.
            instruction (Optional[str], optional): instruction to reranker. Defaults to None.
            sync_client (Optional[httpx.Client], ignored): (ignored) synchronous client to use. Defaults to None.
            async_client (Optional[httpx.AsyncClient], optional): async client to use. Defaults to None.
            max_retry (int, optional): maximum number of retries. Defaults to 30.
            retry_wait (float, optional): wait factor between retries. Defaults to 0.1.
            language (str, optional): prompt language for reranking. Defaults to "en".
            use_mean_score_as_backoff (bool, optional): whether to use mean score as backoff option. Defaults to True.

        Returns:
            Dict[str, float]: mapping document content to score
        """
        if not (self.yes_no_ids and self.yes_no_text):
            raise NotImplementedError("BaseReranker should not be instantiated directly, use its subclasses.")
        if language not in reranker_languages:
            raise ValueError(f"Unsupported language option: {language}, only {reranker_languages} are supported.")
        instruction = f"<Instruct>: {instruction}\n\n" if instruction else ""
        async_client = async_client or httpx.AsyncClient(timeout=self.timeout)
        results: Dict[str, float] = {}
        t = Thread(
            target=self._rerank_worker,
            args=(
                query,
                documents,
                instruction,
                language,
                results,
                async_client,
                max_retry,
                retry_wait,
                use_mean_score_as_backoff,
            ),
        )
        t.start()
        t.join(self.timeout)
        return results

    def invoke(self, messages: List[Dict], tools: Optional[List[Dict]] = None, **kwargs: Dict):
        raise RuntimeError(f"Reranker classes do not support invoke/ainvoke methods.")

    async def ainvoke(self, messages: List[Dict], tools: Optional[List[Dict]] = None, **kwargs: Dict):
        raise RuntimeError(f"Reranker classes do not support invoke/ainvoke methods.")

    def _parse_response(self, response_data: Dict, documents: Optional[List[str]] = None):
        yes_text, no_text = self.yes_no_text
        yes_scores, no_scores = [0], [0]

        # Gather the logprob of returned tokens
        for token in response_data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
            token_text = token["token"].strip().casefold()
            if token_text.startswith(yes_text):
                yes_scores.append(math.exp(token["logprob"]))
            elif token_text.startswith(no_text):
                no_scores.append(math.exp(token["logprob"]))

        confidence = max(yes_scores)
        total_prob = confidence + max(no_scores)
        if total_prob == 0:
            return 0.0
        return confidence / total_prob

    def _request_params(self, **kwargs: Dict) -> Dict:
        messages = kwargs.pop("messages")
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 5,
            "logit_bias": dict.fromkeys(self.yes_no_ids, 5),
            **kwargs,
        }
        match self.preset:
            case "vllm":
                params["chat_template_kwargs"] = {"enable_thinking": False}
            case "aliyun":
                params["extra_body"] = {"enable_thinking": False}
            case _:
                pass

        return params

    def _rerank_worker(
        self,
        query: str,
        documents: List[str],
        instruction: str,
        language: str,
        results: dict,
        async_client: httpx.AsyncClient,
        max_retry: int = 5,
        retry_wait: float = 0.1,
        use_mean_score_as_backoff: bool = True,
    ):
        common_kwargs = dict(instruction=instruction, query=query, yes=self.yes_no_text[0], no=self.yes_no_text[1])
        formatted_templates = (
            TemplateManager().get("rerank_prompt_" + language).format(dict(doc=doc, **common_kwargs)).content
            for doc in documents
        )

        all_query_tasks = asyncio.run(self._run_async_queries(formatted_templates, async_client, max_retry, retry_wait))
        __process_async_rerank_results(documents, all_query_tasks, results, use_mean_score_as_backoff)

    async def _run_async_queries(
        self,
        formatted_templates: Iterable[List[Dict]],
        async_client: httpx.AsyncClient,
        max_retry: int = 5,
        retry_wait: float = 0.1,
    ):
        jobs = (self._async_query_retry(msg, async_client, max_retry, retry_wait) for msg in formatted_templates)
        results = await asyncio.gather(*jobs, return_exceptions=True)
        if hasattr(self, "close"):
            await getattr(self, "close")()
        return results

    async def _async_query_retry(
        self, messages: List, async_client: httpx.AsyncClient, max_retry: int = 5, retry_wait: float = 0.1
    ):
        response = await async_request_with_retry(
            client=async_client,
            max_retry=max_retry,
            retry_wait=retry_wait,
            task="LLM Reranker",
            timeout=self.timeout,
            url=self.api_base,
            json=self._request_params(messages=messages),
            headers=self._request_headers(),
        )
        return self._parse_response(response)


class Qwen3Reranker(BaseReranker):
    """Document relevance re-ranking using Qwen3-Reranker in Chat Completion mode"""

    yes_no_ids: tuple[int, int] = (9693, 2152)
    yes_no_text: tuple[str, str] = ("yes", "no")
    preset: Literal["default", "vllm", "aliyun"] = "aliyun"


class GPT4oReranker(BaseReranker):
    """Document relevance re-ranking using OpenAI models with o200k_base encoding (since GPT-4o)"""

    yes_no_ids: tuple[int, int] = (6763, 1750)
    yes_no_text: tuple[str, str] = ("yes", "no")


class StandardRerankService(BaseReranker):
    """Document relevance re-ranking using rerank API of vLLM / Jina AI / Cohere"""

    def __init__(self, model_name: str, api_key: str, api_base: str, timeout: float, **kwargs):
        """初始化客户端"""
        super().__init__(model_name, api_key, api_base, 0.0, 0.0, timeout=timeout)

    def rerank(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None,
        sync_client: Optional[httpx.Client] = None,
        async_client: Optional[httpx.AsyncClient] = None,
        max_retry: int = 30,
        retry_wait: float = 0.1,
        language: str = "en",
        use_mean_score_as_backoff: bool = True,
        **kwargs,
    ) -> Dict[str, float]:
        """Perform cross-encoder reranking on documents

        Args:
            query (str): query string.
            documents (List[str]): list of documents to rerank.
            instruction (Optional[str], optional): instruction to reranker. Defaults to None.
            sync_client (Optional[httpx.Client], optional): synchronous client to use. Defaults to None.
            async_client (Optional[httpx.AsyncClient], ignored): (ignored) async client to use. Defaults to None.
            max_retry (int, optional): maximum number of retries. Defaults to 30.
            retry_wait (float, optional): wait factor between retries. Defaults to 0.1.
            language (str, optional): prompt language for reranking. Defaults to "en".
            use_mean_score_as_backoff (bool, optional): whether to use mean score as backoff option. Defaults to True.

        Returns:
            Dict[str, float]: mapping document content to score
        """
        sync_client = sync_client or httpx.Client(timeout=self.timeout)
        headers = self._request_headers()
        params = self._request_params(query=query, documents=documents, top_n=len(documents))
        if instruction:
            if not isinstance(instruction, str):
                raise ValueError(f"Invalid reranker instruction: {instruction}")
            params.get("input", params)["instruct"] = instruction

        response = sync_request_with_retry(
            client=sync_client,
            max_retry=max_retry,
            retry_wait=retry_wait,
            task="Reranker",
            timeout=self.timeout,
            url=self.api_base,
            json=params,
            headers=headers,
        )
        return self._parse_response(response, documents=documents)

    def _request_params(self, **kwargs: Dict) -> Dict:
        return dict(model=self.model_name, query=kwargs["query"], documents=kwargs["documents"], top_n=kwargs["top_n"])

    def _parse_response(self, response_data: Dict, documents: Optional[List[str]] = None) -> Dict[str, float]:
        result_dict = dict.fromkeys(documents, 0.0)
        results = response_data.get("output", response_data)["results"]
        for rank_result in results:
            doc = documents[rank_result["index"]]
            result_dict[doc] = rank_result["relevance_score"]
        return result_dict


class AliyunRerankService(StandardRerankService):
    """Document relevance re-ranking using Aliyun's model in Rerank mode"""

    def _request_params(self, **kwargs: Dict) -> Dict:
        documents = kwargs["documents"]
        return {
            "model": self.model_name,
            "input": dict(query=kwargs["query"], documents=documents),
            "parameters": dict(return_documents=False, top_n=kwargs.get("top_n", len(documents))),
        }


def __process_async_rerank_results(
    documents: List[str],
    all_query_tasks: Iterable[Union[BaseException, float]],
    results: Dict,
    use_mean_score_as_backoff: bool = True,
):
    """Process results"""
    all_results = []
    to_fill_with_mean = []
    for doc, result in zip(documents, all_query_tasks):
        if isinstance(result, BaseException):
            to_fill_with_mean.append(doc)
        else:
            results[doc] = result
            all_results.append(result)

    if use_mean_score_as_backoff:
        mean_result = float(np.mean(all_results))
        for doc in to_fill_with_mean:
            results[doc] = mean_result
