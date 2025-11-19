#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import List
import torch
from transformers import AutoModel, AutoTokenizer
from openjiuwen.core.memory.config.config import ModelConfig
import torch.nn.functional as F

class EmbeddingModel:
    def __init__(self, config: ModelConfig):
        self._validate_config(config)
        self.model_config = config
        self.tokenizer, self.model = self._load_model_and_tokenizer()

    def _validate_config(self, config: ModelConfig):
        if config.device not in ["cpu", "cuda"]:
            raise ValueError(f"unsupported device type: {config.device}, only 'cpu' or 'cuda' is supported")

        if not isinstance(config.batch_size, int) or config.batch_size <= 0:
            raise ValueError(f"invalid batch_size: {config.batch_size}，must be a positive integer")

        if not isinstance(config.max_seq_length, int) or config.max_seq_length <= 0:
            raise ValueError(f"invalid max_seq_length: {config.max_seq_length}，must be a positive integer")

    def _load_model_and_tokenizer(self, config: ModelConfig) -> tuple[AutoTokenizer, AutoModel]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name_or_path,
                trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                config.model_name_or_path,
                trust_remote_code=True,
                device_map=config.device
            )
            model.eval()
            return tokenizer, model
        except Exception as e:
            raise RuntimeError(f"embedding model or tokenizer loading failed: {str(e)}") from e

    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        if not texts:
            return []
        all_embeddings: List[List[float]] = []
        device = self.model_config.device

        for i in range(0, len(texts), self.model_config.batch_size):
            batch_texts = texts[i:i + self.model_config.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.model_config.max_seq_length,
                return_tensors="pt", **kwargs
            ).to(device)

            with torch.no_grad():
                model_output = self.model(**inputs)
            embeddings = model_output[0][:, 0]

            if self.model_config.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().tolist())
        return all_embeddings
