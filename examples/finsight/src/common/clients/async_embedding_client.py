from typing import Union, List

from openai import AsyncOpenAI
from openjiuwen.core.retrieval.embedding.api_embedding import EmbeddingConfig, APIEmbedding


class EmbeddingClient:
    def __init__(
            self,
            base_url: str,
            api_key: str,
            model_name: Union[str, List[str]],
            generation_params: dict = None
    ):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.generation_params = generation_params or {}
        self.model_name = model_name

    async def generate_embeddings(
            self, input_texts: List[str],
    ):
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=input_texts
            )
            return [embedding_data.embedding for embedding_data in response.data]
        except Exception as e:
            raise e