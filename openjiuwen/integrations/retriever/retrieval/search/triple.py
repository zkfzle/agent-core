# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator

import numpy as np
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery

from openjiuwen.core.common.logging import logger
from openjiuwen.integrations.retriever.retrieval.search import BaseRetriever



class TripleBeam:
    def __init__(self, nodes: list[TextNode], score: float) -> None:
        self._beam = nodes
        self._exist_triples = {x.text for x in self._beam}
        self._score = score

    def __getitem__(self, idx) -> TextNode:
        return self._beam[idx]

    def __len__(self) -> int:
        return len(self._beam)

    def __contains__(self, triple: TextNode) -> bool:
        return triple.text in self._exist_triples

    def __iter__(self) -> Iterator[TextNode]:
        return iter(self._beam)

    @property
    def triples(self) -> list[TextNode]:
        return self._beam

    @property
    def score(self) -> float:
        return self._score


class TripleBeamSearch:
    def __init__(
        self,
        retriever: BaseRetriever,
        num_beams: int = 10,
        num_candidates_per_beam: int = 100,
        max_length: int = 2,
        encoder_batch_size: int = 256,
    ) -> None:
        if max_length < 1:
            raise ValueError(f"expect max_length >= 1; got {max_length=}")

        self.retriever = retriever
        self.num_beams = num_beams
        self.num_candidates_per_beam = num_candidates_per_beam

        self.max_length = max_length
        self.encoder_batch_size = encoder_batch_size
        self.embed_model = retriever.embed_model

    def __call__(self, query: str, triples: list[TextNode]) -> list[TripleBeam]:
        return asyncio.get_event_loop().run_until_complete(self._beam_search(query, triples))

    @staticmethod
    def _cosine_scores(query_vec: np.ndarray, cand_vecs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query (d,) and candidates (N, d)."""
        q = np.asarray(query_vec, dtype=np.float32)
        c = np.asarray(cand_vecs, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        q_norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
        c_norm = np.linalg.norm(c, axis=1, keepdims=True) + 1e-12
        q = q / q_norm
        c = c / c_norm
        return (q @ c.T).squeeze(0)

    @staticmethod
    def _topk(scores: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        k = min(k, scores.shape[0])
        if k <= 0:
            return [], []
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return idx.tolist(), scores[idx].tolist()

    @staticmethod
    def _format_triple(triple: TextNode) -> str:
        return str(tuple(triple.metadata["triple"]))

    def _format_triples(self, triples: Iterable[TextNode]) -> str:
        return "; ".join(self._format_triple(x) for x in triples)

    async def _beam_search(self, query: str, triples: list[TextNode]) -> list[TripleBeam]:

        if not triples:
            logger.warning("beam search got empty input triples, query=%r", query)
            return []

        texts = [self._format_triple(x) for x in triples] + [query]
        embeddings = self.embed_model.embed_docs(texts, batch_size=self.encoder_batch_size)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        query_embedding = embeddings[-1]  # shape (emb_size,)
        embeddings = embeddings[:-1]  # shape (N, emb_size)

        scores = self._cosine_scores(query_embedding, embeddings)  # shape (N,)
        topk_indices, topk_scores = self._topk(scores, k=self.num_beams)
        beams = [TripleBeam([triples[idx]], score) for idx, score in zip(topk_indices, topk_scores)]

        for _ in range(self.max_length - 1):
            candidates_per_beam = await asyncio.gather(*[self._search_candidates(x) for x in beams])
            beams = self._expand_beams(
                beams=beams,
                candidates_per_beam=candidates_per_beam,
                query_embedding=query_embedding,
            )

        return beams

    def _expand_beams(
        self,
        query_embedding: np.ndarray,
        beams: list[TripleBeam],
        candidates_per_beam: list[list[TextNode]],
    ) -> list[TripleBeam]:
        texts: list[str] = []
        candidate_paths: list[tuple[TripleBeam, TextNode | None]] = []
        exist_triples = {x.text for beam in beams for x in beam}
        for beam, cands in zip(beams, candidates_per_beam):
            if not cands:
                candidate_paths.append((beam, None))
                texts.append(self._format_triples(beam))
                continue

            for triple in cands:
                if triple.text in exist_triples:
                    continue
                candidate_paths.append((beam, triple))
                texts.append(self._format_triples(beam.triples + [triple]))

        if not texts:
            return beams

        embeddings = self.embed_model.embed_docs(texts, batch_size=self.encoder_batch_size)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        next_scores = self._cosine_scores(query_embedding, embeddings)  # shape (N, )

        topk_indices, _ = self._topk(next_scores, k=self.num_beams)

        all_indices = []
        for idx, paths in enumerate(candidate_paths):
            if paths[1] is None:
                all_indices.append(idx)
                continue
            for _ in range(self.num_candidates_per_beam):
                all_indices.append(idx)

        _beams = []
        for idx in topk_indices:
            original_idx = all_indices[idx]
            beam, next_triple = candidate_paths[original_idx]
            if next_triple is None:
                _beams.append(beam)
                continue

            _beams.append(TripleBeam(beam.triples + [next_triple], float(next_scores[original_idx])))

        return _beams

    async def _search_candidates(self, beam: TripleBeam) -> list[TextNode]:
        if len(beam) < 1:
            raise RuntimeError("unexpected empty beam")

        triple = beam[-1].metadata["triple"]

        entities = {triple[0], triple[-1]}
        query_str = " ".join(entities)

        query = VectorStoreQuery(
            query_str=query_str,
            similarity_top_k=self.num_candidates_per_beam,
            mode="text_search",
        )

        nodes = await self.retriever.async_search(query)

        ret = []
        for x in nodes:
            if x in beam:
                continue

            triple = x.metadata["triple"]

            if triple[0] not in entities and triple[-1] not in entities:
                continue

            ret.append(x)

        if not ret:
            logger.warning("empty candidates for beam: %r", self._format_triples(beam))

        return ret
