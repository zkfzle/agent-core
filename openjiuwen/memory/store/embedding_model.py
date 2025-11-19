#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import List
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

class EmbeddingModel:
    BATCH_SIZE = 16
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path
        self.tokenizer, self.model = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> tuple[AutoTokenizer, AutoModel]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
            )
            model.eval()
            return tokenizer, model
        except Exception as e:
            raise RuntimeError(f"embedding model or tokenizer loading failed: {str(e)}") from e

    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        if not texts:
            return []
        all_embeddings: List[List[float]] = []
        device = "cpu"

        for i in range(0, len(texts), self.BATCH_SIZE):
            batch_texts = texts[i:i + self.BATCH_SIZE]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt", **kwargs
            ).to(device)

            with torch.no_grad():
                model_output = self.model(**inputs)
            embeddings = model_output[0][:, 0]
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().tolist())
        return all_embeddings
